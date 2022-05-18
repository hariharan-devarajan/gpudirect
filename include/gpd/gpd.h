//
// Created by Hariharan Devarajan, Hari on 5/17/22.
//

#ifndef GPUDIRECT_GPD_H
#define GPUDIRECT_GPD_H

#include <memory>
#include <fcntl.h>
#include <cassert>
#include <cufile.h>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <atomic>
#include <cuda_runtime.h>
#include <math.h>

namespace gpd {

    struct ClientData {
        CUfileDescr_t _cf_descr;
        void * _device_ptr_base;
        long _buffer_size;
        off_t _dev_ptr_offset;
    };

    class Client {
    private:
        static std::shared_ptr<Client> _instance;
        std::atomic<size_t> _device_ptr_base;
        std::unordered_map<CUfileHandle_t*, ClientData*> _client_data_map;


        Client():_client_data_map(0), _device_ptr_base(0){
            CUfileError_t status = cuFileDriverOpen();
            if (status.err != CU_FILE_SUCCESS) {
                fprintf(stderr, "cuFile driver failed to open\n");
                assert(status.err == CU_FILE_SUCCESS);
            }
        }
    public:
        static std::shared_ptr<Client> Instance() {
            if(_instance == nullptr) {
                _instance = std::make_shared<Client>();
            }
            return _instance;
        }
        CUfileHandle_t Open(const char* filename, int modes, int permissions, long buffer_size) {

            void * dev_ptr_base;
            int fd = open(filename, modes, permissions);
            CUfileDescr_t cf_descr;
            CUfileHandle_t cf_handle;
            if (fd != -1) {
                memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
                cf_descr.handle.fd = fd;
                cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
                CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
                if (status.err != CU_FILE_SUCCESS) {
                    close(fd);
                    fprintf(stderr, "cuFileHandleRegister fd %d status %d\n", fd, status.err);
                    return -1;
                }

                int cuda_result = cudaMalloc(&dev_ptr_base, buffer_size);
                if (cuda_result != CUDA_SUCCESS) {
                    cuFileHandleDeregister(cf_handle);
                    close(fd);
                    fprintf(stderr, "buffer allocation failed %d\n", cuda_result);
                    return -1;
                }
                status = cuFileBufRegister(_device_ptr_base, buffer_size, 0);
                if (status.err != CU_FILE_SUCCESS) {
                    cuFileHandleDeregister(cf_handle);
                    close(fd);
                    cudaFree(_device_ptr_base);
                    fprintf(stderr, "buffer registration failed  %d\n", status.err);
                    return -1;
                }
                _device_ptr_base += buffer_size;
            }
            _client_data_map.erase(cf_handle);
            _client_data_map.insert({cf_handle, {cf_descr, dev_ptr_base, buffer_size, 0}});
        }

        ssize_t Write(CUfileHandle_t cf_handle, const void* data, size_t size, off_t file_offset) {
            auto iter = _client_data_map.find(&cf_handle);
            ssize_t ret = -1;
            if (iter != _client_data_map.end()) {
                off_t current_offset = 0;
                size_t num_pieces = ceil(size / iter._buffer_size);
                for (int i = 0; i < num_pieces; ++i) {
                    size_t write_size = size % iter._buffer_size;
                    if (write_size == 0) write_size = iter._buffer_size;
                    CudaError_t status = cudaMemcpy(iter._device_ptr_base, data, write_size);
                    ret += cuFileWrite(cf_handle, iter._device_ptr_base + current_offset, write_size,
                                       file_offset + current_offset, iter._dev_ptr_offset + current_offset);
                    current_offset += write_size;
                }
            }
            if (ret < 0 || ret != size) {
                fprintf(stderr, "cuFileWrite failed  %d\n", ret);
            }
            return ret;
        }

        ssize_t Read(CUfileHandle_t cf_handle, void* data, size_t size, off_t file_offset) {
            auto iter = _client_data_map.find(&cf_handle);
            ssize_t ret = -1;
            if (iter != _client_data_map.end()) {
                off_t current_offset = 0;
                size_t num_pieces = ceil(size / iter._buffer_size);
                for (int i = 0; i < num_pieces; ++i) {
                    size_t read_size = size % iter._buffer_size;
                    if (read_size == 0) read_size = iter._buffer_size;
                    ret += cuFileRead(cf_handle, iter._device_ptr_base + current_offset, read_size,
                                       file_offset + current_offset, iter._dev_ptr_offset + current_offset);
                    CudaError_t status = cudaMemcpy(data, iter._device_ptr_base, read_size);
                    current_offset += read_size;
                }
            }
            if (ret < 0 || ret != size) {
                fprintf(stderr, "cuFileRead failed  %d\n", ret);
            }
            return ret;
        }

        ssize_t Close(CUfileHandle_t cf_handle) {
            auto iter = _client_data_map.find(&cf_handle);
            ssize_t ret = -1;
            if (iter != _client_data_map.end()) {
                CUfileError_t status = cuFileBufDeregister(iter._device_ptr_base);
                if (status.err != CU_FILE_SUCCESS) {
                    cudaFree(iter._device_ptr_base);
                    (void) cuFileHandleDeregister(cf_handle);
                    close(iter._cf_descr.handle.fd);
                    fprintf(stderr, "buffer deregister failed\n");
                    return -1;
                }
                cudaFree(iter._device_ptr_base);
                (void) cuFileHandleDeregister(cf_handle);
                close(iter._cf_descr.handle.fd);
            }
        }
        ~Client() {
            (void) cuFileDriverClose();
        }
    };
}


#endif //GPUDIRECT_GPD_H
