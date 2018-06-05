#ifndef DRAY_DATA_LOGGER_HPP
#define DRAY_DATA_LOGGER_HPP

#include <dray/utils/yaml_writer.hpp>

namespace dray 
{

class DataLogger 
{
public:
  ~DataLogger();
  static DataLogger *get_instance();
  void open(const std::string &entry_name);
  void close();

  void add_value(const std::string &value);
  
  template<typename T>
  void add_entry(const std::string key, const T &value)
  {
    m_writer.add_entry(key, value);
  }

  void write_log(std::string filename);
protected:
  DataLogger();
  DataLogger(DataLogger const &);
  YamlWriter m_writer;
  static class DataLogger* m_instance;
};

#define DRAY_LOG_OPEN(name) dray::DataLogger::get_instance()->open(name);
#define DRAY_LOG_CLOSE() dray::DataLogger::get_instance()->close();
#define DRAY_LOG_ENTRY(key,value) dray::DataLogger::get_instance()->add_entry(key,value);
#define DRAY_LOG_VALUE(value) dray::DataLogger::get_instance()->add_value(value);
#define DRAY_LOG_WRITE(file_name) dray::DataLogger::get_instance()->write_log(file_name);

} // namspace dray
#endif
