#ifndef DRAY_REDISTRIBUTE_HPP
#define DRAY_REDISTRIBUTE_HPP

#include <dray/collection.hpp>

namespace dray
{



class Redistribute
{
protected:

  struct CommInfo
  {
    int32 m_dest_rank;
    int32 m_domain_id;
    int32 m_src_rank;
    int32 m_src_idx; // local domain to send (only used by sender)
    static bool compare(const CommInfo &a, const CommInfo &b)
    {
      return a.m_domain_id < b.m_domain_id;
    }
  };

  std::vector<CommInfo> m_comm_info;

public:
  Redistribute();

  Collection execute(Collection &collection,
                     const std::vector<int32> &src_list,
                     const std::vector<int32> &dest_list);
protected:

  void build_schedule(Collection &collection,
                      const std::vector<int32> &src_list,
                      const std::vector<int32> &dest_list);

  void send_recv_metadata(Collection &collection);
  void send_recv(Collection &collection);
  std::map<int32, DataSet> m_recv_q;
};

};//namespace dray

#endif// header gaurd
