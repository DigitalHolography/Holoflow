#include "holonp/sum.hh"
#include "holonp/std.hh"
#include "holonp/var.hh"
#include "holonp/argmin.hh"
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <numeric>
#include <stdexcept>
#include "curaii/cuda.hh"
namespace holonp {
namespace { void axes_json(nlohmann::json&j,const std::vector<int>&a,bool k){j["keepdims"]=k;if(a.empty())j["axis"]=nullptr;else if(a.size()==1)j["axis"]=a[0];else j["axis"]=a;} void axes_parse(const nlohmann::json&j,std::vector<int>&a,bool&k){a.clear();k=j.value("keepdims",false);if(!j.contains("axis")||j["axis"].is_null())return;if(j["axis"].is_number_integer())a={j["axis"].get<int>()};else j.at("axis").get_to(a);}
struct R{std::vector<int>a;bool k;}; std::vector<size_t> shape(const holoflow::core::TDesc&d,const R&s){std::vector<bool>m(d.rank());for(auto x:s.a){if(x<0)x+=static_cast<int>(d.rank());if(x<0||x>=static_cast<int>(d.rank()))throw std::invalid_argument("reduction axis out of range");m[static_cast<size_t>(x)]=true;}std::vector<size_t>o=d.shape;if(s.k){for(size_t i=0;i<o.size();++i)if(m[i])o[i]=1;}else{for(size_t i=o.size();i-->0;)if(m[i])o.erase(o.begin()+i);}return o;}
template<int Mode> __global__ void red(const float*in,float*out,size_t n){if(blockIdx.x||threadIdx.x)return;float x=0; if(Mode==0){for(size_t i=0;i<n;++i)x+=in[i];out[0]=x;}else{float mean=0;for(size_t i=0;i<n;++i)mean+=in[i];mean/=n;for(size_t i=0;i<n;++i){float d=in[i]-mean;x+=d*d;}out[0]=Mode==1?sqrtf(x/n):x/n;}}
template<int Mode> __global__ void amin(const float*in,unsigned short*out,size_t n){if(blockIdx.x||threadIdx.x)return;size_t p=0;for(size_t i=1;i<n;++i)if(in[i]<in[p])p=i;out[0]=(unsigned short)p;}
class Task:public holoflow::core::ISyncTask{public:Task(int m,cudaStream_t s,size_t n):m(m),s(s),n(n){}holoflow::core::OpResult execute(holoflow::core::SyncCtx&c)override{if(m==3)amin<0><<<1,1,0,s>>>((float*)c.inputs[0].data(),(unsigned short*)c.outputs[0].data(),n);else if(m==0)red<0><<<1,1,0,s>>>((float*)c.inputs[0].data(),(float*)c.outputs[0].data(),n);else if(m==1)red<1><<<1,1,0,s>>>((float*)c.inputs[0].data(),(float*)c.outputs[0].data(),n);else red<2><<<1,1,0,s>>>((float*)c.inputs[0].data(),(float*)c.outputs[0].data(),n);CUDA_CHECK(cudaGetLastError());return holoflow::core::OpResult::Ok;}private:int m;cudaStream_t s;size_t n;};
template<class S>void tj(nlohmann::json&j,const S&s){axes_json(j,s.axis,s.keepdims);}template<class S>void fj(const nlohmann::json&j,S&s){axes_parse(j,s.axis,s.keepdims);}template<class S>holoflow::core::InferResult inf(std::span<const holoflow::core::TDesc>x,const nlohmann::json&j,holoflow::core::DType dt){if(x.size()!=1||x[0].dtype!=holoflow::core::DType::F32||x[0].num_elements()==0)throw std::invalid_argument("reduction requires non-empty F32 input");S s=j.get<S>();return {{x[0]},{{shape(x[0],{s.axis,s.keepdims}),dt,holoflow::core::MemLoc::Device}},{},{false},{false},holoflow::core::TaskKind::Sync};}template<class S>std::unique_ptr<holoflow::core::ISyncTask> cr(std::span<const holoflow::core::TDesc>x,const nlohmann::json&j,const holoflow::core::SyncCreateCtx&c,int m){inf<S>(x,j, m==3?holoflow::core::DType::U16:holoflow::core::DType::F32);return std::make_unique<Task>(m,c.stream,x[0].num_elements());}
}
void to_json(nlohmann::json&j,const SumSettings&s){tj(j,s);}void from_json(const nlohmann::json&j,SumSettings&s){fj(j,s);}
#define RIMPL(N,S,D,M) holoflow::core::InferResult N##Factory::infer(std::span<const holoflow::core::TDesc>x,const nlohmann::json&j)const{return inf<S>(x,j,D);}std::unique_ptr<holoflow::core::ISyncTask>N##Factory::create(std::span<const holoflow::core::TDesc>x,const nlohmann::json&j,const holoflow::core::SyncCreateCtx&c)const{return cr<S>(x,j,c,M);}std::unique_ptr<holoflow::core::ISyncTask>N##Factory::update(std::unique_ptr<holoflow::core::ISyncTask>,std::span<const holoflow::core::TDesc>x,const nlohmann::json&j,const holoflow::core::SyncCreateCtx&c)const{return cr<S>(x,j,c,M);}
RIMPL(Sum,SumSettings,holoflow::core::DType::F32,0)RIMPL(Std,StdSettings,holoflow::core::DType::F32,1)RIMPL(Var,VarSettings,holoflow::core::DType::F32,2)RIMPL(Argmin,ArgminSettings,holoflow::core::DType::U16,3)
}
