#ifndef WP_PARSER_C_H_
#define WP_PARSER_C_H_

#include "wp_parser_y.h"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuPrint.H>
#include <iostream>
#include <cassert>
#include <set>
#include <string>
#include <type_traits>

struct wp_parser* wp_c_parser_new (char const* function_body);

#define MAX_DEPTH 8

template <int Depth, std::enable_if_t<(Depth<MAX_DEPTH), int> = 0>
AMREX_GPU_HOST_DEVICE
__attribute__((noinline))
double
wp_ast_eval (struct wp_node* node, double const* x)
{
    double result;

    switch (node->type)
    {
    case WP_NUMBER:
    {
        result = ((struct wp_number*)node)->value;
        break;
    }
    case WP_SYMBOL:
    {
#if AMREX_DEVICE_COMPILE
        int i =((struct wp_symbol*)node)->ip.i;
        result = x[i];
#else
        result = *(((struct wp_symbol*)node)->ip.p);
#endif
        break;
    }
    case WP_ADD:
    {
        result = wp_ast_eval<Depth+1>(node->l,x) + wp_ast_eval<Depth+1>(node->r,x);
        break;
    }
    case WP_SUB:
    {
        result = wp_ast_eval<Depth+1>(node->l,x) - wp_ast_eval<Depth+1>(node->r,x);
        break;
    }
    case WP_MUL:
    {
        result = wp_ast_eval<Depth+1>(node->l,x) * wp_ast_eval<Depth+1>(node->r,x);
        break;
    }
    case WP_DIV:
    {
        result = wp_ast_eval<Depth+1>(node->l,x) / wp_ast_eval<Depth+1>(node->r,x);
        break;
    }
    case WP_NEG:
    {
        result = -wp_ast_eval<Depth+1>(node->l,x);
        break;
    }
    case WP_F1:
    {
        result = wp_call_f1(((struct wp_f1*)node)->ftype,
                wp_ast_eval<Depth+1>(((struct wp_f1*)node)->l,x));
        break;
    }
    case WP_F2:
    {
        result = wp_call_f2(((struct wp_f2*)node)->ftype,
                wp_ast_eval<Depth+1>(((struct wp_f2*)node)->l,x),
                wp_ast_eval<Depth+1>(((struct wp_f2*)node)->r,x));
        break;
    }
    case WP_ADD_VP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->rip.i;
        result = node->lvp.v + x[i];
#else
        result = node->lvp.v + *(node->rip.p);
#endif
        break;
    }
    case WP_ADD_PP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->lvp.ip.i;
        int j = node->rip.i;
        result = x[i] + x[j];
#else
        result = *(node->lvp.ip.p) + *(node->rip.p);
#endif
        break;
    }
    case WP_SUB_VP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->rip.i;
        result = node->lvp.v - x[i];
#else
        result = node->lvp.v - *(node->rip.p);
#endif
        break;
    }
    case WP_SUB_PP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->lvp.ip.i;
        int j = node->rip.i;
        result = x[i] - x[j];
#else
        result = *(node->lvp.ip.p) - *(node->rip.p);
#endif
        break;
    }
    case WP_MUL_VP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->rip.i;
        result = node->lvp.v * x[i];
#else
        result = node->lvp.v * *(node->rip.p);
#endif
        break;
    }
    case WP_MUL_PP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->lvp.ip.i;
        int j = node->rip.i;
        result = x[i] * x[j];
#else
        result = *(node->lvp.ip.p) * *(node->rip.p);
#endif
        break;
    }
    case WP_DIV_VP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->rip.i;
        result = node->lvp.v / x[i];
#else
        result = node->lvp.v / *(node->rip.p);
#endif
        break;
    }
    case WP_DIV_PP:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->lvp.ip.i;
        int j = node->rip.i;
        result = x[i] / x[j];
#else
        result = *(node->lvp.ip.p) / *(node->rip.p);
#endif
        break;
    }
    case WP_NEG_P:
    {
#if AMREX_DEVICE_COMPILE
        int i = node->rip.i;
        result = -x[i];
#else
        result = -*(node->lvp.ip.p);
#endif
        break;
    }
    default:
#if AMREX_DEVICE_COMPILE
        AMREX_DEVICE_PRINTF("wp_ast_eval: unknown node type %d\n", node->type);
#else
        std::cout << "wp_ast_eval: unknown node type " << node->type << "\n";
#endif
    }

    return result;
}

template <int Depth, std::enable_if_t<Depth == MAX_DEPTH,int> = 0>
AMREX_GPU_HOST_DEVICE
__attribute__((noinline))
double
wp_ast_eval (struct wp_node* node, double const* x)
{
#if AMREX_DEVICE_COMPILE
    AMREX_DEVICE_PRINTF("wp_ast_eval: MAX_DEPTH %d not big enough\n",
                        MAX_DEPTH);
#else
    std::cout << "wp_ast_eval: MAX_DEPTH" << MAX_DEPTH
                      << "not big enough\n";
#endif
    return 0.;
}

inline
void
wp_ast_get_symbols (struct wp_node* node, std::set<std::string>& symbols)
{
    switch (node->type)
    {
    case WP_NUMBER:
        break;
    case WP_SYMBOL:
        symbols.emplace(((struct wp_symbol*)node)->name);
        break;
    case WP_ADD:
    case WP_SUB:
    case WP_MUL:
    case WP_DIV:
    case WP_ADD_PP:
    case WP_SUB_PP:
    case WP_MUL_PP:
    case WP_DIV_PP:
        wp_ast_get_symbols(node->l, symbols);
        wp_ast_get_symbols(node->r, symbols);
        break;
    case WP_NEG:
    case WP_NEG_P:
        wp_ast_get_symbols(node->l, symbols);
        break;
    case WP_F1:
        wp_ast_get_symbols(((struct wp_f1*)node)->l, symbols);
        break;
    case WP_F2:
        wp_ast_get_symbols(((struct wp_f2*)node)->l, symbols);
        wp_ast_get_symbols(((struct wp_f2*)node)->r, symbols);
        break;
    case WP_ADD_VP:
    case WP_SUB_VP:
    case WP_MUL_VP:
    case WP_DIV_VP:
        wp_ast_get_symbols(node->r, symbols);
        break;
    default:
        std::cout << "wp_ast_get_symbols: unknown node type " << node->type << "\n";
    }
}

#endif
