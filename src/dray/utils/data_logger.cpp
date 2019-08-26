#include <dray/utils/data_logger.hpp>

#include <fstream>
#include <iostream>

namespace dray
{

Logger* Logger::m_instance  = NULL;

Logger::Logger()
{
  std::stringstream log_name;
  log_name<<"dray";
  log_name<<".log";
  m_stream.open(log_name.str().c_str(), std::ofstream::out);
  if(!m_stream.is_open())
    std::cout<<"Warning: could not open the devil ray log file\n";
}

Logger::~Logger()
{
  if(m_stream.is_open())
    m_stream.close();
}

Logger* Logger::get_instance()
{
  if(m_instance == NULL)
    m_instance =  new Logger();
  return m_instance;
}

std::ofstream& Logger::get_stream()
{
  return m_stream;
}

void
Logger::write(const int level, const std::string &message, const char *file, int line)
{
  if(level == 0)
    m_stream<<"<Info> \n";
  else if (level == 1)
    m_stream<<"<Warning> \n";
  else if (level == 2)
    m_stream<<"<Error> \n";
  m_stream<<"  message: "<<message<<" \n  file: "<<file<<" \n  line: "<<line<<"\n";
}

/* ----------------------------------------------------------------------------------*/

DataLogger* DataLogger::m_instance= NULL;

DataLogger::DataLogger()
{
}

DataLogger::~DataLogger()
{

}

DataLogger*
DataLogger::get_instance()
{
  if(DataLogger::m_instance== NULL)
  {
    DataLogger::m_instance = new DataLogger();
  }
  return DataLogger::m_instance;
}

void DataLogger::add_value(const std::string &value)
{
  m_writer.add_value(value);
}

void
DataLogger::write_log(std::string filename)
{
  std::stringstream log_name;
  std::ofstream stream;
  log_name<<filename<<".yaml";
  stream.open(log_name.str().c_str(), std::ofstream::out);
  if(!stream.is_open())
  {
    std::cerr<<"Warning: could not open the dray data log file '"<<filename<<"'\n";
    return;
  }
  stream<<m_writer.get_stream().str();
  stream.close();
}

void
DataLogger::open(const std::string &entry_name)
{
  m_writer.start_block(entry_name);
}
void
DataLogger::close()
{
  m_writer.end_block();
}

} // namespace dray
