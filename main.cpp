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
#include <errno.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <gpd/gpd.h>

//#include "cufile_sample_utils.h"
using namespace std;

int main(void) {
  using namespace gpd;
  char *testfn = getenv("TESTFILE");
  if (testfn == NULL) {
    std::cerr << "No testfile defined via TESTFILE.  Exiting." << std::endl;
    return -1;
  }
  off_t file_offset = 0;
  off_t devPtr_offset = 0;
  ssize_t IO_size = 1UL << 24;
  size_t buff_size = IO_size;
  auto client = Client::Instance();
  auto handle = client->Open(testfn, O_CREAT|O_WRONLY|O_DIRECT, 0644, buff_size);
  auto cuda_buf = client->GetDeviceBuffer(handle);
  char write_char = 'w';
  cudaMemset((void *) cuda_buf, write_char, buff_size);
  client->Write(handle, IO_size, file_offset);
  cuda_buf = client->GetDeviceBuffer(handle);
  cudaMemset((void *) cuda_buf, 'r', buff_size);
  client->Read(handle, IO_size, file_offset);
  for (int i=0;i<buff_size;++i) {
    assert(*((char*)cuda_buf + i) == write_char);
  }
  client->Close(handle);
  return 0;
}