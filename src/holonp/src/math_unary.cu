#include "holonp/math_unary.hh"
#include "holonp/clip.hh"
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include "curaii/cuda.hh"
namespace holonp {
void to_json(nlohmann::json&j,const UnaryMathSettings&){j=nlohmann::json::object();} void from_json(const nlohmann::json&,UnaryMathSettings&){}
void to_json(nlohmann::json&j,const ClipSettings&s){j={{"min",s.min},{"max",s.max}};} void from_json(const nlohmann::json&j,ClipSettings&s){s.min=j.value("min",0.f);s.max=j.value("max",1.f);if(s.min>s.max)throw std::invalid_argument("Clip: min must not exceed max");}
namespace { enum class Op{Sqrt,Real,Imag,Angle,Log,Finite,Clip};
__device__ float apply(float x,Op o,float lo,float hi){switch(o){case Op::Sqrt:return sqrtf(x);case Op::Real:return x;case Op::Imag:return 0;case Op::Angle:return x>=0?0.f:3.14159265358979323846f;case Op::Log:return logf(x);case Op::Finite:return isfinite(x)?1.f:0.f;case Op::Clip:return fminf(hi,fmaxf(lo,x));}return 0;}
__device__ float apply(cuFloatComplex x,Op o,float lo,float hi){switch(o){case Op::Real:return x.x;case Op::Imag:return x.y;case Op::Angle:return atan2f(x.y,x.x);case Op::Finite:return isfinite(x.x)&&isfinite(x.y)?1.f:0.f;default:return apply(x.x,o,lo,hi);}}
template<class T>__global__ void kernel(const T*in,float*out,size_t n,Op o,float lo,float hi){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=apply(in[i],o,lo,hi);}
class Task:public holoflow::core::ISyncTask{public:Task(Op o,holoflow::core::TDesc d,cudaStream_t s,float l,float h):o_(o),d_(std::move(d)),s_(s),l_(l),h_(h){}holoflow::core::OpResult execute(holoflow::core::SyncCtx&c)override{auto n=d_.num_elements();int b=256,g=(int)((n+b-1)/b);if(d_.dtype==holoflow::core::DType::CF32)kernel<<<g,b,0,s_>>>((cuFloatComplex*)c.inputs[0].data(),(float*)c.outputs[0].data(),n,o_,l_,h_);else kernel<<<g,b,0,s_>>>((float*)c.inputs[0].data(),(float*)c.outputs[0].data(),n,o_,l_,h_);CUDA_CHECK(cudaGetLastError());return holoflow::core::OpResult::Ok;}private:Op o_;holoflow::core::TDesc d_;cudaStream_t s_;float l_,h_;};
holoflow::core::InferResult inf(std::span<const holoflow::core::TDesc>i,Op){if(i.size()!=1||i[0].mem_loc!=holoflow::core::MemLoc::Device||i[0].num_elements()==0||(i[0].dtype!=holoflow::core::DType::F32&&i[0].dtype!=holoflow::core::DType::CF32))throw std::invalid_argument("holonp unary: invalid input");return {.input_descs={i[0]},.output_descs={{i[0].shape,holoflow::core::DType::F32,holoflow::core::MemLoc::Device}},.owned_inputs={false},.owned_outputs={false},.kind=holoflow::core::TaskKind::Sync};}
template<Op O>holoflow::core::InferResult it(std::span<const holoflow::core::TDesc>i,const nlohmann::json&){return inf(i,O);}template<Op O>std::unique_ptr<holoflow::core::ISyncTask>ct(std::span<const holoflow::core::TDesc>i,const nlohmann::json&j,const holoflow::core::SyncCreateCtx&c){it<O>(i,j);float l=0,h=1;if constexpr(O==Op::Clip){auto x=j.get<ClipSettings>();l=x.min;h=x.max;}return std::make_unique<Task>(O,i[0],c.stream,l,h);}
#define I(N,O) holoflow::core::InferResult N##Factory::infer(std::span<const holoflow::core::TDesc>i,const nlohmann::json&j)const{return it<O>(i,j);}std::unique_ptr<holoflow::core::ISyncTask>N##Factory::create(std::span<const holoflow::core::TDesc>i,const nlohmann::json&j,const holoflow::core::SyncCreateCtx&c)const{return ct<O>(i,j,c);}std::unique_ptr<holoflow::core::ISyncTask>N##Factory::update(std::unique_ptr<holoflow::core::ISyncTask>,std::span<const holoflow::core::TDesc>i,const nlohmann::json&j,const holoflow::core::SyncCreateCtx&c)const{return ct<O>(i,j,c);}
} I(Sqrt,Op::Sqrt) I(Real,Op::Real) I(Imag,Op::Imag) I(Angle,Op::Angle) I(Log,Op::Log) I(Isfinite,Op::Finite) I(Clip,Op::Clip)
}
