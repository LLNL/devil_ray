#ifndef DRAY_DATA_LOGGER_HPP
#define DRAY_DATA_LOGGER_HPP

#include <dray/utils/yaml_writer.hpp>
#include <fstream>

namespace dray
{

class Logger
{
public:
  ~Logger();
  static Logger *get_instance();
  void write(const int level, const std::string &message, const char *file, int line);
  std::ofstream & get_stream();
protected:
  Logger();
  Logger(Logger const &);
  std::ofstream m_stream;
  static class Logger* m_instance;
};

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

#ifdef DRAY_ENABLE_LOGGING
#define DRAY_INFO(msg) dray::Logger::get_instance()->get_stream() <<"<Info>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define DRAY_WARN(msg) dray::Logger::get_instance()->get_stream() <<"<Warn>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define DRAY_ERROR(msg) dray::Logger::get_instance()->get_stream() <<"<Error>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;

#define DRAY_LOG_OPEN(name) dray::DataLogger::get_instance()->open(name);
#define DRAY_LOG_CLOSE() dray::DataLogger::get_instance()->close();
#define DRAY_LOG_ENTRY(key,value) dray::DataLogger::get_instance()->add_entry(key,value);
#define DRAY_LOG_VALUE(value) dray::DataLogger::get_instance()->add_value(value);
#define DRAY_LOG_WRITE(file_name) dray::DataLogger::get_instance()->write_log(file_name);

#else
#define DRAY_INFO(msg)
#define DRAY_WARN(msg)
#define DRAY_ERROR(msg)

#define DRAY_LOG_OPEN(name)
#define DRAY_LOG_CLOSE()
#define DRAY_LOG_ENTRY(key,value)
#define DRAY_LOG_VALUE(value)
#define DRAY_LOG_WRITE(file_name)
#endif

} // namspace dray
#endif
