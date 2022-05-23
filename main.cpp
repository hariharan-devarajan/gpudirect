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
#include <libunwind.h>
#include <elfutils/libdwfl.h>
#include <cxxabi.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <signal.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <gpd/gpd.h>
/** Print a demangled stack backtrace of the caller function to FILE* out. */
inline void debugInfo(std::stringstream &out,const void* ip)
{

  char *debuginfo_path=NULL;

  Dwfl_Callbacks callbacks={
      .find_elf=dwfl_linux_proc_find_elf,
      .find_debuginfo=dwfl_standard_find_debuginfo,
      .section_address=NULL,
      .debuginfo_path=&debuginfo_path,
  };

  Dwfl* dwfl=dwfl_begin(&callbacks);
  assert(dwfl!=NULL);

  assert(dwfl_linux_proc_report (dwfl, getpid())==0);
  assert(dwfl_report_end (dwfl, NULL, NULL)==0);

  Dwarf_Addr addr = (uintptr_t)ip;

  Dwfl_Module* module=dwfl_addrmodule (dwfl, addr);

  const char* function_name = dwfl_module_addrname(module, addr);
  out << function_name << "(";

  Dwfl_Line *line=dwfl_getsrc (dwfl, addr);
  if(line!=NULL)
  {
    int nline;
    Dwarf_Addr addr;
    const char* filename=dwfl_lineinfo (line, &addr,&nline,NULL,NULL,NULL);

    out << function_name << strrchr(filename,'/')+1 << ":" << nline;
  }
  else
  {
    out << function_name << ip;
  }
}

inline void printStackTrace(int skip,int sig)
{
  unw_context_t uc;
  unw_getcontext(&uc);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &uc);
  std::stringstream ss;
  while(unw_step(&cursor)>0)
  {

    unw_word_t ip;
    unw_get_reg(&cursor, UNW_REG_IP, &ip);

    unw_word_t offset;
    char name[32];
    //assert(unw_get_proc_name(&cursor, name,sizeof(name), &offset)==0);

    if(skip<=0)
    {
      if (unw_get_proc_name(&cursor, name,sizeof(name), &offset)==0) {
        ss << "\tat ";
        debugInfo(ss, (void *) (ip - 4));
        ss << ")\n";
      } else {
        break;
      }
    }

    if(strcmp(name,"main")==0)
      break;

    skip--;

  }
  int pid = getpid();
  fprintf(stderr, "[ERROR]: Error: signal %d on Process %d:\n%s\n", sig, pid, ss.str().c_str());
}

inline void handler(int sig) {
  printStackTrace(0, sig);
  exit(0);
}
//#include "cufile_sample_utils.h"
using namespace std;

int main(void) {
  signal(SIGSEGV, handler);
  signal(SIGABRT, handler);
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
  auto handle = client->Open(testfn, O_CREAT|O_RDWR|O_DIRECT, 0644, buff_size);
  auto cuda_buf = client->GetDeviceBuffer(handle);
  char write_char = 'w';
  cudaMemset((void *) cuda_buf, write_char, buff_size);
  client->Write(handle, IO_size, file_offset);
  cuda_buf = client->GetDeviceBuffer(handle);
  cudaMemset((void *) cuda_buf, 'r', buff_size);
  client->Read(handle, IO_size, file_offset);
  cuda_buf = client->GetDeviceBuffer(handle);
  client->Close(handle);
  client->Finalize();
  return 0;
}