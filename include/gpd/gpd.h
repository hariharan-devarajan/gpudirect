//
// Created by Hariharan Devarajan, Hari on 5/17/22.
//

#ifndef GPUDIRECT_GPD_H
#define GPUDIRECT_GPD_H

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>

bool operator==(const CUfileDescr_t o1, const CUfileDescr_t o2) {
  return o1.handle.fd == o2.handle.fd && o1.type == o2.type;
}
namespace gpd {

struct ClientData {
  CUfileDescr_t _cf_descr;
  void* _device_ptr_base;
  long _buffer_size;
  off_t _dev_ptr_offset;
  ClientData()
      : _cf_descr(),
        _device_ptr_base(nullptr),
        _buffer_size(),
        _dev_ptr_offset() {}
  ClientData(const ClientData& other)
      : _cf_descr(other._cf_descr),
        _device_ptr_base(other._device_ptr_base),
        _buffer_size(other._buffer_size),
        _dev_ptr_offset(other._dev_ptr_offset) {}
  ClientData(const ClientData&& other)
      : _cf_descr(other._cf_descr),
        _device_ptr_base(other._device_ptr_base),
        _buffer_size(other._buffer_size),
        _dev_ptr_offset(other._dev_ptr_offset) {}
  ClientData& operator=(const ClientData& other) {
    _cf_descr = other._cf_descr;
    _device_ptr_base = other._device_ptr_base;
    _buffer_size = other._buffer_size;
    _dev_ptr_offset = other._dev_ptr_offset;
    return *this;
  }
  bool operator==(const ClientData& other) const {
    return _cf_descr == other._cf_descr &&
           _device_ptr_base == other._device_ptr_base &&
           _buffer_size == other._buffer_size &&
           _dev_ptr_offset == other._dev_ptr_offset;
  }
};

class Client {
 private:
  static std::shared_ptr<Client> _instance;
  std::unordered_map<CUfileHandle_t, ClientData*> _client_data_map;
 public:
  Client() : _client_data_map(0) {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
      fprintf(stderr, "cuFile driver failed to open\n");
      assert(status.err == CU_FILE_SUCCESS);
    }
  }
  static std::shared_ptr<Client> Instance() {
    if (_instance == nullptr) {
      _instance = std::make_shared<Client>();
    }
    return _instance;
  }
  CUfileHandle_t Open(const char* filename, int modes, int permissions,
                      long buffer_size) {
    void* dev_ptr_base;
    int fd = open(filename, modes, permissions);
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    if (fd != -1) {
      memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
      cf_descr.handle.fd = fd;
      cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
      if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        fprintf(stderr, "cuFileHandleRegister fd %d status %d\n", fd,
                status.err);
        return cf_handle;
      }

      int cuda_result = cudaMalloc(&dev_ptr_base, buffer_size);
      if (cuda_result != CUDA_SUCCESS) {
        cuFileHandleDeregister(cf_handle);
        close(fd);
        fprintf(stderr, "buffer allocation failed %d\n", cuda_result);
        return cf_handle;
      }
      status = cuFileBufRegister(dev_ptr_base, buffer_size, 0);
      if (status.err != CU_FILE_SUCCESS) {
        cuFileHandleDeregister(cf_handle);
        close(fd);
        cudaFree(dev_ptr_base);
        fprintf(stderr, "buffer registration failed  %d\n", status.err);
        return cf_handle;
      }
      _client_data_map.erase(cf_handle);
      auto client_data = new ClientData();
      client_data->_cf_descr = cf_descr;
      client_data->_buffer_size = buffer_size;
      client_data->_dev_ptr_offset = 0;
      client_data->_device_ptr_base = dev_ptr_base;
      _client_data_map.insert({cf_handle, client_data});
    }
    return cf_handle;
  }

  void* GetDeviceBuffer(CUfileHandle_t cf_handle) {
    auto iter = _client_data_map.find(cf_handle);
    if (iter != _client_data_map.end()) {
      return iter->second->_device_ptr_base;
    }
    return NULL;
  }


  ssize_t Write(CUfileHandle_t cf_handle, size_t size, off_t file_offset) {
    auto iter = _client_data_map.find(cf_handle);
    ssize_t ret = 0;
    if (iter != _client_data_map.end()) {
      off_t current_offset = 0;
      size_t num_pieces = ceil(size / iter->second->_buffer_size);
      for (int i = 0; i < num_pieces; ++i) {
        size_t write_size = size % iter->second->_buffer_size;
        if (write_size == 0) write_size = iter->second->_buffer_size;
        ret += cuFileWrite(cf_handle,
                           (char*)iter->second->_device_ptr_base + current_offset,
                           write_size, file_offset + current_offset,
                           iter->second->_dev_ptr_offset + current_offset);
        current_offset += write_size;
      }
    }
    if (ret < 0 || ret != size) {
      fprintf(stderr, "cuFileWrite failed  %d should be %d\n", ret, size);
    } else {
      //fprintf(stdout, "cuFileWrite success with bytes %d\n", ret);
    }
    return ret;
  }

  ssize_t Read(CUfileHandle_t cf_handle, size_t size, off_t file_offset) {
    auto iter = _client_data_map.find(cf_handle);
    ssize_t ret = 0;
    if (iter != _client_data_map.end()) {
      off_t current_offset = 0;
      size_t num_pieces = ceil(size / iter->second->_buffer_size);
      for (int i = 0; i < num_pieces; ++i) {
        size_t read_size = size % iter->second->_buffer_size;
        if (read_size == 0) read_size = iter->second->_buffer_size;
        ret += cuFileRead(cf_handle,
                          (char*)iter->second->_device_ptr_base + current_offset,
                          read_size, file_offset + current_offset,
                          iter->second->_dev_ptr_offset + current_offset);
//        auto status =
//            cudaMemcpy(data, iter->second->_device_ptr_base, read_size,
//                       cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        current_offset += read_size;
      }
    }
    if (ret < 0 || ret != size) {
      fprintf(stderr, "cuFileRead failed %d should be %d\n", ret, size);
    } else {
      //fprintf(stdout, "cuFileRead success with bytes %d\n", ret);
    }
    return ret;
  }

  ssize_t Close(CUfileHandle_t cf_handle) {
    auto iter = _client_data_map.find(cf_handle);
    ssize_t ret = -1;
    if (iter != _client_data_map.end()) {
      CUfileError_t status =
          cuFileBufDeregister(iter->second->_device_ptr_base);
      if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "buffer deregister failed\n");
        cudaFree(iter->second->_device_ptr_base);
        (void)cuFileHandleDeregister(cf_handle);
        close(iter->second->_cf_descr.handle.fd);
        return -1;
      }
      cudaFree(iter->second->_device_ptr_base);
      (void)cuFileHandleDeregister(cf_handle);
      close(iter->second->_cf_descr.handle.fd);
      delete (iter->second);
      //fprintf(stdout, "buffer deregister success\n");
    }
    return ret;
  }
  void Finalize() {
    (void)cuFileDriverClose();
  }
};
}  // namespace gpd

#endif  // GPUDIRECT_GPD_H
