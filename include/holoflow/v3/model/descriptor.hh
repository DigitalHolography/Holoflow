#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace holoflow::model {

struct DescriptorNodeProperties {
  std::string id;
  std::string type;
  nlohmann::json config;
};

struct DescriptorEdgeProperties {};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                              DescriptorNodeProperties,
                              DescriptorEdgeProperties>
    DescriptorGraph;

typedef boost::graph_traits<DescriptorGraph>::vertex_descriptor
    DescriptorVertex;
typedef boost::graph_traits<DescriptorGraph>::edge_descriptor DescriptorEdge;

struct DescriptorNodePropertyWriter {
  DescriptorNodePropertyWriter(const DescriptorGraph &g) : graph(g) {}

  template <typename VertexT>
  void operator()(std::ostream &out, const VertexT &v) const {
    std::string config_str = graph[v].config.dump(2);
    config_str = escape_graphviz_label(config_str);

    out << "[label=\"" << graph[v].id << "\\n(" << graph[v].type << ")\\n"
        << config_str << "\"]";
  }

private:
  const DescriptorGraph &graph;

  std::string escape_graphviz_label(const std::string &s) const {
    std::ostringstream oss;
    for (char c : s) {
      if (c == '"' || c == '\\')
        oss << '\\';
      oss << c;
    }
    return oss.str();
  }
};

struct DescriptorEdgePropertyWriter {
  template <typename Edge>
  void operator()(std::ostream &out, const Edge &) const {
    out << "";
  }
};

struct DescriptorWriter {
  void operator()(std::ostream &out) const { out << "rankdir=LR;\n"; }
};

} // namespace holoflow::model
