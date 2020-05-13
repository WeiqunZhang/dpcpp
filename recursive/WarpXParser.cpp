#include "WarpXParser.H"
#include <algorithm>

WarpXParser::WarpXParser (std::string const& func_body)
{
    define(func_body);
}

void
WarpXParser::define (std::string const& func_body)
{
    clear();

    m_expression = func_body;
    m_expression.erase(std::remove(m_expression.begin(),m_expression.end(),'\n'),
                       m_expression.end());
    std::string f = m_expression + "\n";

    m_parser = wp_c_parser_new(f.c_str());
}

WarpXParser::~WarpXParser ()
{
    clear();
}

void
WarpXParser::clear ()
{
    m_expression.clear();
    m_varnames.clear();

    if (m_parser) wp_parser_delete(m_parser);
    m_parser = nullptr;
}

void
WarpXParser::registerVariable (std::string const& name, double& var)
{
    wp_parser_regvar(m_parser, name.c_str(), &var);
    m_varnames.push_back(name);
}

void
WarpXParser::registerVariables (std::vector<std::string> const& names)
{
    for (int j = 0; j < names.size(); ++j) {
        wp_parser_regvar(m_parser, names[j].c_str(), &(m_variables[j]));
        m_varnames.push_back(names[j]);
    }
}

void
WarpXParser::setConstant (std::string const& name, double c)
{
    wp_parser_setconst(m_parser, name.c_str(), c);
}

void
WarpXParser::print () const
{
    wp_ast_print(m_parser->ast);
}

std::string const&
WarpXParser::expr () const
{
    return m_expression;
}

std::set<std::string>
WarpXParser::symbols () const
{
    std::set<std::string> results;
    wp_ast_get_symbols(m_parser->ast, results);
    return results;
}
