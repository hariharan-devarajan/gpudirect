// To compile this sample code:
//
// nvcc gds_helloworld.cxx -o gds_helloworld -lcufile
//
// Set the environment variable TESTFILE
// to specify the name of the file on a GDS enabled filesystem
//
// Ex:   TESTFILE=/mnt/gds/gds_test ./gds_helloworld
//
//
#include <fcntl.h>
#include <gpd/gpd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "gpd/gpd.h"


class Timer {
 public:
  Timer() : elapsed_time(0) {}
  void resumeTime() { t1 = std::chrono::high_resolution_clock::now(); }
  double pauseTime() {
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed_time += std::chrono::duration<double>(t2 - t1).count();
    return elapsed_time;
  }
  double getElapsedTime() { return elapsed_time; }

 private:
  std::chrono::high_resolution_clock::time_point t1;
  double elapsed_time;
};
//#include "cufile_sample_utils.h"
using namespace std;

int main(int argc, char* argv[]) {
  using namespace gpd;
  if (argc < 5) {
    std::cerr << "Pass gpudirect <mode: 0 posix and 1 gpudirect><filename> <transfer-size-kb> <num-ops>." << std::endl;
    return -1;
  }
  int mode = atoi(argv[1]);
  char * test_filename = argv[2];
  size_t transfer_size_kb = atol(argv[3]);
  size_t num_ops = atol(argv[4]);
  ssize_t io_size_bytes = transfer_size_kb * 1024;
  remove(test_filename);
  /**
   * Validation
   */
  if (mode == 0) {
    fprintf(stdout, "Running Posix Mode with filename %s, transfer size %ld KB, and number of operations %ld.\n", test_filename,
            transfer_size_kb, num_ops);
  } else if (mode == 1) {
    fprintf(stdout, "Running GPUDirect Mode with filename %s, transfer size %ld KB, and number of operations %ld.\n", test_filename,
            transfer_size_kb, num_ops);
  } else {
    fprintf(stderr, "Unknown Mode %d. Use 0 for posix and 1 for gpudirect.\n");
    return -1;
  }
  Timer initialize, memory, write_t, finalize;
  if (mode == 0) {
    initialize.resumeTime();
    auto client = Client::Instance();
    int fd = open(test_filename, O_WRONLY| O_CREAT | O_SYNC, 0600);
    initialize.pauseTime();
    assert(fd != -1);
    initialize.resumeTime();
    void* dev_ptr_base;
    int cuda_result = cudaMalloc(&dev_ptr_base, io_size_bytes);
    if (cuda_result != CUDA_SUCCESS) {
      close(fd);
      fprintf(stderr, "buffer allocation failed %d\n", cuda_result);
      return -1;
    }
    CUfileError_t status = cuFileBufRegister(dev_ptr_base, io_size_bytes, 0);
    if (status.err != CU_FILE_SUCCESS) {
      close(fd);
      cudaFree(dev_ptr_base);
      fprintf(stderr, "buffer registration failed  %d\n", status.err);
      return -1;
    }
    initialize.pauseTime();
    memory.resumeTime();
    char write_char = 'w';
    cudaMemset((void *) dev_ptr_base, write_char, io_size_bytes);
    auto write_data = std::vector<char>(io_size_bytes, write_char);
    memory.pauseTime();
    for(int i = 0; i < num_ops; ++i) {
      memory.resumeTime();
      cudaMemcpy(write_data.data(),dev_ptr_base,io_size_bytes,cudaMemcpyDeviceToHost);
      memory.pauseTime();
      write_t.resumeTime();
      auto written = write(fd, write_data.data(), io_size_bytes);
      write_t.pauseTime();
      if(written != io_size_bytes) {
        fprintf(stderr, "Write not successful %d %d %s\n", written, errno, strerror(errno));
      }
      assert(written == io_size_bytes);
    }
    finalize.resumeTime();
    close(fd);
    finalize.pauseTime();
  } else if (mode == 1) {
    initialize.resumeTime();
    auto client = Client::Instance();
    auto handle = client->Open(test_filename, O_WRONLY|O_CREAT|O_DIRECT, 0600, io_size_bytes);
    initialize.pauseTime();
    memory.resumeTime();
    auto cuda_buf = client->GetDeviceBuffer(handle);
    char write_char = 'w';
    cudaMemset((void *) cuda_buf, write_char, io_size_bytes);
    memory.pauseTime();
    for(int i = 0; i < num_ops; ++i) {
      off_t file_offset = i * transfer_size_kb * 1024;
      write_t.resumeTime();
      auto written = client->Write(handle, io_size_bytes, file_offset);
      write_t.pauseTime();
      assert(written == io_size_bytes);
    }
    finalize.resumeTime();
    client->Close(handle);
    client->Finalize();
    finalize.pauseTime();
  }
  remove(test_filename);
  if (mode == 0 && transfer_size_kb == 1 && num_ops == 1)
    fprintf(stdout, "\bTiming,Mode,transfer_size_kb,num_ops,init,memory,write,finalize\n");
  char* mode_str;
  if (mode == 0) mode_str = "POSIX";
  if (mode == 1) mode_str = "GPUDIRECT";
  fprintf(stdout, "\bTiming,%s,%ld,%ld,%f,%f,%f,%f\n", mode_str, transfer_size_kb, num_ops,
          initialize.getElapsedTime(), memory.getElapsedTime(), write_t.getElapsedTime(),
          finalize.getElapsedTime());
  return 0;
}
