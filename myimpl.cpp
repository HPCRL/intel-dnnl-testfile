/**
   dst ( n, oc, oh, ow) = bias(oc) + \sum_ic \sum_kh \sum_kw src(n, ic, oh*SH+kh-ph0, ow*SW+kw-pw0) * weights(oc, ic, kh, kw)   
 *
 */



#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>
#include <assert.h>
#include "dnnl_debug.h"
#include "example_utils.hpp"
#include "omp.h"
using namespace dnnl;
using namespace std;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim),
            std::multiplies<memory::dim>());
}

void test_conv(engine::kind engine_kind){
    using tag = memory::format_tag;
    engine eng(engine_kind, 0);
    stream engine_stream(eng);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
//    dst(b, f, x, y) = in(b, c, x+w, y+s) * ker(f,c,w,s);
    
    //size:
//    b = 256, x=y=224,112,56,28,14,7  (f=64 c=3), (f=64,c=64) (128,64),(128,128), (256,128),(256,256), w=s=3 or 7

    const int B = 1, W = 3, S = 3, X = 28, Y = 28, F = 512, C = 256;
    const int stride_ob = F*X*Y;
    const int stride_of = X*Y;
    const int stride_ox = Y;
    const int stride_oy = 1;

    const int stride_ib = C*(X+W-1)*(Y+S-1);
    const int stride_ic = (X+W-1)*(Y+S-1);
    const int stride_ix = (Y+S-1);
    const int stride_iy = 1;

    const int stride_kf = C*W*S;
    const int stride_kc = W*S;
    const int stride_kw = S;
    const int stride_ks = 1;

    auto o_offset = [=](int b, int f, int x, int y){
                        return b*stride_ob + f*stride_of + x*stride_ox + y*stride_oy;
                    };

    auto i_offset = [=](int b, int c, int x, int y){
                        return b*stride_ib + c*stride_ic + x*stride_ix + y*stride_iy;
                    };

    auto k_offset = [=](int f, int c, int w, int s){
                        return f*stride_kf + c*stride_kc + w*stride_kw + s*stride_ks;
                    };



    std::vector<float> output(B*F*X*Y);
    std::vector<float> input(B*C*(X+W-1)*(Y+S-1));
    std::vector<float> kernel(F*C*W*S);
//    std::vector<float> conv_bias(F);
    memory::dims conv_strides = {1,1};
    memory::dims conv_padding = {0,0};
//    memory::dims conv_bias_tz = {F};    
    

    for(int b = 0; b<B; b++)
    for(int c = 0; c<C; c++)
    for(int x = 0; x<X+W-1; x++)
    for(int y = 0; y<Y+S-1; y++)
    {
        int ioff = i_offset(b,c,x,y);
        input[ioff] = -std::cos(ioff / 10.0f);
    }

    for(int f = 0; f<F; f++)
    for(int w = 0; w<W; w++)
    for(int c = 0; c<C; c++)
    for(int s = 0; s<S; s++)
    {
        int koff = k_offset(f,c,w,s);
        input[koff] = -std::cos(koff/10.0f);
    }



    auto input_md = memory::desc(
        {B, C, (X+W-1), (Y+S-1)},
        memory::data_type::f32,
        tag::any//{stride_ib, stride_ic, stride_ix, stride_iy}
        );
    
    auto output_md = memory::desc(
        {B, F, X, Y},
        memory::data_type::f32,
        tag::any//{stride_ob, stride_of, stride_ox, stride_oy}
        );

    auto kernel_md = memory::desc(
        {F, C, W, S},
        memory::data_type::f32,
        tag::any//{stride_kf, stride_kc, stride_kw, stride_ks}
        );

//    auto bias_md = memory::desc({conv_bias_tz}, memory::data_type::f32, tag::any);
    // auto input_mem = memory(input_md, eng);
    // auto output_mem = memory(output_md, eng);
    // auto kernel_mem = memory(kernel_md, eng);

    auto user_input_mem = memory({ {B, C, X+W-1, Y+S-1}, memory::data_type::f32, tag::nchw }, eng);
    auto user_kernel_mem = memory({ {F, C, W, S}, memory::data_type::f32, tag::oihw }, eng);
//    auto user_bias_mem = memory({{conv_bias_tz }, memory::data_type::f32, tag::x}, eng);
    cout<<"start omp wtime\n";
    double start = omp_get_wtime();    
    write_to_dnnl_memory(input.data(), user_input_mem);
    write_to_dnnl_memory(kernel.data(), user_kernel_mem);
//    write_to_dnnl_memory(conv_bias.data(), user_bias_mem);





    


    
    std::cout<<"conv desc create\n";

//    auto conv_desc =
//        convolution_forward::desc(prop_kind::forward_inference,
//                                  algorithm::convolution_direct, input_md, kernel_md, bias_md,
//                                  output_md, conv_strides, conv_padding, conv_padding);
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, input_md, kernel_md,
             output_md, conv_strides, conv_padding,
            conv_padding);
        
    std::cout<<"conv prim desc create\n";
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

    auto conv_input_mem = user_input_mem;
    if(conv_prim_desc.src_desc() != user_input_mem.get_desc()){
        conv_input_mem = memory(conv_prim_desc.src_desc(), eng );
        net.push_back(reorder(user_input_mem, conv_input_mem));
        net_args.push_back(
            { {DNNL_ARG_FROM, user_input_mem}, {DNNL_ARG_TO, conv_input_mem} }
            );
    }

    auto conv_kernel_mem = user_kernel_mem;
    if(conv_prim_desc.weights_desc() != user_kernel_mem.get_desc()){
        conv_kernel_mem = memory(conv_prim_desc.weights_desc(), eng);
        net.push_back(reorder(user_kernel_mem, conv_kernel_mem));
        net_args.push_back(
            {{DNNL_ARG_FROM, user_kernel_mem}, {DNNL_ARG_TO, conv_kernel_mem}}
            );
    }

    auto conv_output_mem = memory(conv_prim_desc.dst_desc(), eng);

    net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back(
        { {DNNL_ARG_SRC, conv_input_mem},
          {DNNL_ARG_WEIGHTS, conv_kernel_mem},
//          {DNNL_ARG_WEIGHTS, user_bias_mem},
          {DNNL_ARG_DST, conv_output_mem}
        });

    assert(net.size() == net_args.size() && "something is missing, net size != net arg size\n");
    std::cout<<"start!\n";
    double start2 = omp_get_wtime();
    int totiter = 10;
    for(int iter = 0; iter< totiter; iter++ ){
    for(size_t i = 0; i < net.size(); i++){
        net.at(i).
            execute(
                s, net_args.at(i));
    }
//    std::cout<<"before wait!\n";
    s.wait();
    }
    double end = omp_get_wtime();
    double flop_cnt = 2.0*B*W*S*X*Y*C*F*totiter;
    std::cout<<"fin\n";
    std::cout<<"all time = "<<end-start<<endl;
    std::cout<<"exe time = "<<end-start2<<endl;
    std::cout<<"all time flop = "<<flop_cnt/(end-start)/1000.0/1000.0/1000.0<<endl;
    std::cout<<"exe time flop = "<<flop_cnt/(end-start2)/1000.0/1000.0/1000.0<<endl;

}


int main(int argc, char**argv){
    try{
        engine:: kind engine_kind = parse_engine_kind(argc, argv);
        test_conv(engine_kind);
        std::cout<<" conv pass\n";        
    }catch(dnnl::error &e){
        std::cerr << "DNNL error: " << e.what() << std::endl
                  << "Error status: " << dnnl_status2str(e.status) << std::endl;
        return 1;
    } catch (std::string &e) {
        std::cerr << "Error in the example: " << e << std::endl;
        return 2;
    }

    return 0;    
    
}
