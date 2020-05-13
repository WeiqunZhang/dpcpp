#ifndef WARPX_PARSER_H_
#define WARPX_PARSER_H_

#include <array>
#include <vector>
#include <string>
#include <set>

#include "wp_parser_c.h"
#include "wp_parser_y.h"

template <int N> class GpuParser;

class WarpXParser
{
public:
    WarpXParser (std::string const& func_body);
    WarpXParser () = default;
    ~WarpXParser ();
    void define (std::string const& func_body);

    void setConstant (std::string const& name, double c);

    //
    // Option 1: Register every variable to an address provided.
    //           Assign values to external variables.
    //           Call eval().
    void registerVariable (std::string const& name, double& var);
    //
    inline double eval () const noexcept;

    //
    // Option 2: Register all variables at once. Parser will create
    //               variables internally.
    //           Call eval(...) with variable values.
    void registerVariables (std::vector<std::string> const& names);
    //
    template <typename T, typename... Ts> inline
    double eval (T x, Ts... yz) const noexcept;

    void print () const;

    std::string const& expr () const;

    std::set<std::string> symbols () const;

    template <int N> friend class GpuParser;

private:
    void clear ();

    template <typename T> inline
    void unpack (double* p, T x) const noexcept;

    template <typename T, typename... Ts> inline
    void unpack (double* p, T x, Ts... yz) const noexcept;

    std::string m_expression;
    struct wp_parser* m_parser = nullptr;
    mutable std::array<double,16> m_variables;
    mutable std::vector<std::string> m_varnames;
};

inline
double
WarpXParser::eval () const noexcept
{
    return wp_ast_eval(m_parser->ast,nullptr);
}

template <typename T, typename... Ts>
inline
double
WarpXParser::eval (T x, Ts... yz) const noexcept
{
    unpack(m_variables.data(), x, yz...);
    return eval();
}

template <typename T>
inline
void
WarpXParser::unpack (double* p, T x) const noexcept
{
    *p = x;
}

template <typename T, typename... Ts>
inline
void
WarpXParser::unpack (double* p, T x, Ts... yz) const noexcept
{
    *p++ = x;
    unpack(p, yz...);
}

#endif
