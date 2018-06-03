#include <dray/utils/data_logger.hpp>

#include <fstream>
#include <iostream>

namespace dray {

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
    std::cerr<<"Warning: could not open the dray data log file '"<<filename<"'\n";
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
