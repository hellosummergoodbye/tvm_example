#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <fstream>
#include <iostream>


int main()
  {
  tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("../source/module.so");

  int dtype_code  = kDLFloat;
  int dtype_bits  = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id   = 0;

  std::ifstream json_in("../source/deploy.json", std::ios::in);
  std::string   json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());

  json_in.close();

  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

  DLTensor* x;
  DLTensor* y;
  DLTensor* z;

  int     in_ndim    =  2;
  int64_t in_shape[] = {2,2};

  int     out_ndim    =  2;
  int64_t out_shape[] = {2,2};

  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

  std::cout<<"Input X matrix 2x2 size"<<std::endl;

  std::cin>>*(static_cast<float*>((x->data))+0)>>*(static_cast<float*>((x->data))+1);
  std::cin>>*(static_cast<float*>((x->data))+2)>>*(static_cast<float*>((x->data))+3);

  std::cout<<"Input Y matrix 2x2 size"<<std::endl;

  std::cin>>*(static_cast<float*>((y->data))+0)>>*(static_cast<float*>((y->data))+1);
  std::cin>>*(static_cast<float*>((y->data))+2)>>*(static_cast<float*>((y->data))+3);


  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");

  set_input("x", x);
  set_input("y", y);


  tvm::runtime::PackedFunc run = mod.GetFunction("run");

  run();

  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &z);

  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

  get_output(0, z);

  auto z_output = static_cast<float*>(z->data);

  std::cout<<"X * Y is"<<std::endl;

  std::cout<<z_output[0]<<' '<<z_output[1]<<std::endl;
  std::cout<<z_output[2]<<' '<<z_output[3]<<std::endl;

  TVMArrayFree(x);
  TVMArrayFree(y);
  TVMArrayFree(z);

  return 0;
  }
