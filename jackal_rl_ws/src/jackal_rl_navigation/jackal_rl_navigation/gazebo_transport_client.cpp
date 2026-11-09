// One transport Node per environment. Tabs frame local stdin/stdout only;
// requests on Gazebo Transport remain the repository's protobuf messages.
#include <ignition/transport/Node.hh>
#include <ignition/msgs/boolean.pb.h>
#include <ignition/msgs/world_control.pb.h>
#include <ignition/msgs/pose.pb.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <string>
#include <vector>

template <typename Request>
std::string Send(ignition::transport::Node &node, const std::string &service,
                 unsigned int timeout, const std::string &text) {
  Request request;
  if (!google::protobuf::TextFormat::ParseFromString(text, &request))
    return "BAD_REQUEST";
  ignition::msgs::Boolean reply;
  bool result = false;
  if (!node.Request(service, request, timeout, reply, result)) return "TIMEOUT";
  if (!result) return "CALL_FAILED";
  return reply.data() ? "OK_TRUE" : "OK_FALSE";
}

int main() {
  ignition::transport::Node node;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::vector<std::string> fields;
    size_t begin = 0;
    for (int i = 0; i < 4; ++i) {
      auto end = line.find('\t', begin);
      if (end == std::string::npos) break;
      fields.push_back(line.substr(begin, end - begin));
      begin = end + 1;
    }
    if (fields.size() != 4) return 2;
    fields.push_back(line.substr(begin));
    std::string status;
    try {
      auto timeout = static_cast<unsigned int>(std::stoul(fields[1]));
      if (fields[3] == "ignition.msgs.WorldControl")
        status = Send<ignition::msgs::WorldControl>(node, fields[2], timeout, fields[4]);
      else if (fields[3] == "ignition.msgs.Pose")
        status = Send<ignition::msgs::Pose>(node, fields[2], timeout, fields[4]);
      else status = "BAD_REQUEST";
    } catch (const std::exception &error) {
      std::cerr << error.what() << std::endl;
      status = "CLIENT_ERROR";
    }
    std::cout << fields[0] << '\t' << status << std::endl;
  }
}
