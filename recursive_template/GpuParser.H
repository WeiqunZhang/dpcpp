#ifndef WARPX_GPU_PARSER_H_
#define WARPX_GPU_PARSER_H_

#include "WarpXParser.H"
#include "AMReX_GpuQualifiers.H"
#include <type_traits>
#include <cstdlib>

void* myMalloc (std::size_t);
void myFree (void*);

namespace amrex {
    template <class T, std::size_t N>
    struct GpuArray
    {
        AMREX_GPU_HOST_DEVICE inline
        const T& operator [] (int i) const noexcept { return arr[i]; }

        AMREX_GPU_HOST_DEVICE inline
        T& operator [] (int i) noexcept { return arr[i]; }

        AMREX_GPU_HOST_DEVICE inline
        const T* data() const noexcept { return arr; };

        AMREX_GPU_HOST_DEVICE inline
        std::size_t size() const noexcept { return N; };

        AMREX_GPU_HOST_DEVICE inline
        const T* begin() const noexcept { return arr; };

        AMREX_GPU_HOST_DEVICE inline
        const T* end() const noexcept { return arr + N; };

        AMREX_GPU_HOST_DEVICE inline
        T* begin() noexcept { return arr; };

        AMREX_GPU_HOST_DEVICE inline
        T* end() noexcept { return arr + N; };

        T arr[N];
    };

    template <typename T, typename U1, typename... Us>
    struct Same;

    template <typename T, typename U>
    struct Same<T,U>
    {
        static constexpr bool value = std::is_same<T,U>::value;
    };

    template <typename T, typename U1, typename... Us>
    struct Same
    {
        static constexpr bool value = std::is_same<T,U1>::value and Same<T,Us...>::value;
    };
}


// When compiled for CPU, wrap WarpXParser and enable threading.
// When compiled for GPU, store one copy of the parser in
// CUDA managed memory for __device__ code, and one copy of the parser
// in CUDA managed memory for __host__ code. This way, the parser can be
// efficiently called from both host and device.
template <int N>
class GpuParser
{
public:
    GpuParser (WarpXParser const& wp);
    void clear ();

    template <typename... Ts>
    AMREX_GPU_HOST_DEVICE
    std::enable_if_t<sizeof...(Ts) == N
                     and amrex::Same<double,Ts...>::value,
                     double>
    operator() (Ts... var) const noexcept
    {
        amrex::GpuArray<double,N> l_var{var...};
#if AMREX_DEVICE_COMPILE
// WarpX compiled for GPU, function compiled for __device__
        return wp_ast_eval<0>(m_gpu_parser.ast, l_var.data());
#else
// WarpX compiled for GPU, function compiled for __host__
        return wp_ast_eval<0>(m_cpu_parser->ast, nullptr);
#endif
    }

private:

    // Copy of the parser running on __device__
    struct wp_parser m_gpu_parser;
    // Copy of the parser running on __host__
    struct wp_parser* m_cpu_parser;
    mutable amrex::GpuArray<double,N> m_var;
};

template <int N>
GpuParser<N>::GpuParser (WarpXParser const& wp)
{
    struct wp_parser* a_wp = wp.m_parser;
    // Initialize GPU parser: allocate memory in CUDA managed memory,
    // copy all data needed on GPU to m_gpu_parser
    m_gpu_parser.sz_mempool = wp_ast_size(a_wp->ast);
    m_gpu_parser.p_root = (struct wp_node*)myMalloc(m_gpu_parser.sz_mempool);
    m_gpu_parser.p_free = m_gpu_parser.p_root;
    // 0: don't free the source
    m_gpu_parser.ast = wp_parser_ast_dup(&m_gpu_parser, a_wp->ast, 0);
    for (int i = 0; i < N; ++i) {
        wp_parser_regvar_gpu(&m_gpu_parser, wp.m_varnames[i].c_str(), i);
    }

    // Initialize CPU parser:
    m_cpu_parser = wp_parser_dup(a_wp);
    for (int i = 0; i < N; ++i) {
        wp_parser_regvar(m_cpu_parser, wp.m_varnames[i].c_str(), &m_var[i]);
    }
}


template <int N>
void
GpuParser<N>::clear ()
{
    myFree(m_gpu_parser.ast);
    wp_parser_delete(m_cpu_parser);
}

#endif
