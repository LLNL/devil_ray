#include <dray/filters/redistribute.hpp>
#include <dray/dray_node_to_dataset.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>

#include <conduit.hpp>
#include <algorithm>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>

#define DRAY_CHECK_MPI_ERROR( check_mpi_err_code )                  \
{                                                                   \
    if( static_cast<int>(check_mpi_err_code) != MPI_SUCCESS)        \
    {                                                               \
        char check_mpi_err_str_buff[MPI_MAX_ERROR_STRING];          \
        int  check_mpi_err_str_len=0;                               \
        MPI_Error_string( check_mpi_err_code ,                      \
                         check_mpi_err_str_buff,                    \
                         &check_mpi_err_str_len);                   \
                                                                    \
        DRAY_ERROR("MPI call failed: \n"                            \
                      << " error code = "                           \
                      <<  check_mpi_err_code  << "\n"               \
                      << " error message = "                        \
                      <<  check_mpi_err_str_buff << "\n");          \
    }                                                               \
}

#endif

namespace dray
{

namespace detail
{

void pack_grid_function(conduit::Node &n_gf,
                        std::vector<std::pair<size_t,unsigned char*>> &gf_ptrs)
{
  size_t values_bytes = n_gf["values"].total_bytes_compact();
  unsigned char * values_ptr = (unsigned char*)n_gf["values"].data_ptr();

  std::pair<size_t, unsigned char*> value_pair;
  value_pair.first = values_bytes;
  value_pair.second = values_ptr;

  gf_ptrs.push_back(value_pair);

  size_t conn_bytes = n_gf["conn"].total_bytes_compact();
  unsigned char * conn_ptr = (unsigned char*)n_gf["conn"].data_ptr();

  std::pair<size_t, unsigned char*> conn_pair;
  conn_pair.first = conn_bytes;
  conn_pair.second = conn_ptr;

  gf_ptrs.push_back(conn_pair);
}

void pack_dataset(conduit::Node &n_dataset,
                  std::vector<std::pair<size_t,unsigned char*>> &gf_ptrs)
{
  pack_grid_function(n_dataset["topology/grid_function"], gf_ptrs);
  const int32 num_fields = n_dataset["fields"].number_of_children();
  for(int32 i = 0; i < num_fields; ++i)
  {
    conduit::Node &field = n_dataset["fields"].child(i);
    pack_grid_function(field["grid_function"], gf_ptrs);
  }
}

void strip_helper(conduit::Node &node)
{
  const int32 num_children = node.number_of_children();

  if(num_children == 0)
  {
    if(node.name() == "conn" || node.name() == "values")
    {
      node.reset();
    }
  }

  for(int32 i = 0; i < num_children; ++i)
  {
    strip_helper(node.child(i));
  }
}

void strip_arrays(const conduit::Node &input, conduit::Node &output)
{
  output.set_external(input);
  strip_helper(output);
}

DataSet dataset_from_meta(const conduit::Node &meta)
{

}

}//namespace detail



Redistribute::Redistribute()
{
}


Collection
Redistribute::execute(Collection &collection,
                      const std::vector<int32> &src_list,
                      const std::vector<int32> &dest_list)
{
  Collection res;
  build_schedule(collection, src_list, dest_list);
  send_recv_metadata(collection);
  send_recv(collection);
  return res;
}


void
Redistribute::build_schedule(Collection &collection,
                             const std::vector<int32> &src_list,
                             const std::vector<int32> &dest_list)
{
  const int32 total_domains = collection.size();
  if(src_list.size() != total_domains)
  {
    DRAY_ERROR("src list needs to be of size total domains");
  }
#ifdef DRAY_MPI_ENABLED

  int32 local_size = collection.local_size();
  int32 rank, procs;

  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &procs);

  std::vector<int32> dom_counts;
  dom_counts.resize(procs);

  MPI_Allgather(&local_size, 1, MPI_INT, &dom_counts[0], 1, MPI_INT, comm);

  std::vector<int32> dom_offsets;
  dom_offsets.resize(procs);
  dom_offsets[0] = 0;

  for(int i = 1; i < procs; ++i)
  {
    dom_offsets[i] = dom_offsets[i-1] + dom_counts[i-1];
  }

  m_comm_info.clear();

  const int32 list_size = src_list.size();

  // figure out sends
  const int32 rank_offset = dom_offsets[rank];
  int32 dom_count = dom_counts[rank];
  for(int32 i = 0; i < dom_count; ++i)
  {
    const int32 index = i + rank_offset;
    if(dest_list[index] != rank)
    {
      CommInfo send;
      send.m_src_idx = i;
      send.m_src_rank = rank;
      send.m_dest_rank = dest_list[index];
      send.m_domain_id = index;
      m_comm_info.push_back(send);
    }
  }

  // figure out recvs
  for(int32 i = 0; i < list_size; ++i)
  {
    if(dest_list[i] == rank && src_list[i] != rank)
    {
      CommInfo recv;
      recv.m_src_rank = src_list[i];
      recv.m_dest_rank = rank;
      recv.m_domain_id = i;
      m_comm_info.push_back(recv);
    }
  }

  // in order to not-deadlock, we will use domain_id
  // as a global ordering. Interleave sends and recvs based
  // on this ordering

  struct CompareCommInfo
  {
    bool operator()(const CommInfo &a, const CommInfo &b) const
    {
      return a.m_domain_id < b.m_domain_id;
    }
  };

  std::sort(m_comm_info.begin(), m_comm_info.end(), CompareCommInfo());

  if(rank == 0) std::cout<<"Send recv schedule\n";
  for(int32 i = 0; i < procs; ++i)
  {
    if(rank == i)
    {
      std::cout<<"**** Rank "<<i<<" ***\n";
      for(int a = 0; a < m_comm_info.size(); ++a)
      {
        if(m_comm_info[a].m_dest_rank != rank)
        {
          std::cout<<"Send idx "<<m_comm_info[a].m_domain_id
                   <<" ->  "<<m_comm_info[a].m_dest_rank<<"\n";
        }
        else
        {
          std::cout<<"Recv idx "<<m_comm_info[a].m_domain_id<<" <-  "<<m_comm_info[a].m_src_rank<<"\n";
        }
      }
    }
    MPI_Barrier(comm);
  }
#endif

}


void Redistribute::send_recv_metadata(Collection &collection)
{
  // we will send and recv everything but the actual arrays
  // so we can allocated space and recv the data via Isend/Irecv

#ifdef DRAY_MPI_ENABLED
  const int32 total_comm = m_comm_info.size();

  int32 rank = dray::mpi_rank();
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());

  for(int32 i = 0; i < total_comm; ++i)
  {
    CommInfo info = m_comm_info[i];
    bool send = info.m_src_rank == rank;

    if(send)
    {
      conduit::Node n_domain;
      DataSet domain = collection.domain(info.m_src_idx);
      domain.to_node(n_domain);
      conduit::Node meta;
      detail::strip_arrays(n_domain,meta);
      conduit::relay::mpi::send_using_schema(meta,
                                             info.m_dest_rank,
                                             info.m_domain_id,
                                             comm);
    }
    else
    {

      conduit::Node n_domain_meta;
      conduit::relay::mpi::recv_using_schema(n_domain_meta,
                                             info.m_src_rank,
                                             info.m_domain_id,
                                             comm);
      //if(rank == 0) n_domain_meta.print();
      // allocate a data set to for recvs
      m_recv_q[info.m_domain_id] = to_dataset(n_domain_meta);
      if(info.m_domain_id == 0)
      {
        //n_domain_meta.schema().print();
        std::cout<<"***** "<<n_domain_meta["fields"].number_of_children()<<"\n";
        std::cout<<"***** "<<m_recv_q[info.m_domain_id].field_info()<<"\n";

      }
    }

  }
#endif
}

void Redistribute::send_recv(Collection &collection)
{

#ifdef DRAY_MPI_ENABLED
  const int32 total_comm = m_comm_info.size();

  int32 rank = dray::mpi_rank();
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  int32 send_count = 0;
  int32 recv_count = 0;
  for(int32 i = 0; i < total_comm; ++i)
  {
    CommInfo info = m_comm_info[i];
    bool send = info.m_src_rank == rank;

    if(send)
    {
      send_count++;
    }
    else
    {
      recv_count++;
    }
  }

  // get all the pointers together

  // tag/vector< size, ptr>
  std::map<int32,std::vector<std::pair<size_t,unsigned char*>>> send_buffers;
  std::map<int32, int32> send_dests;
  std::map<int32,std::vector<std::pair<size_t,unsigned char*>>> recv_buffers;
  std::map<int32, int32> recv_srcs;
  for(int32 i = 0; i < total_comm; ++i)
  {
    CommInfo info = m_comm_info[i];
    bool send = info.m_src_rank == rank;
    const int32 base_tag = info.m_domain_id * 1000;
    // we don't need to keep the conduit nodes around
    // since they point directly to dray memory
    if(send)
    {
      conduit::Node n_domain;
      DataSet domain = collection.domain(info.m_src_idx);
      domain.to_node(n_domain);
      detail::pack_dataset(n_domain, send_buffers[base_tag]);
      send_dests[base_tag] = info.m_dest_rank;
    }
    else
    {
      DataSet &domain = m_recv_q[info.m_domain_id];
      conduit::Node n_domain;
      domain.to_node(n_domain);
      detail::pack_dataset(n_domain, recv_buffers[base_tag]);
      recv_srcs[base_tag] = info.m_src_rank;
    }
  }

  std::vector<MPI_Request> requests;

  // send it
  for(auto &domain : send_buffers)
  {
    const int32 base_tag = domain.first;
    int32 tag_counter = 0;
    // TODO: check for max int size
    for(auto &buffer : domain.second)
    {
      MPI_Request request;
      int32 mpi_error = MPI_Isend(buffer.second,
                                  static_cast<int>(buffer.first),
                                  MPI_BYTE,
                                  send_dests[base_tag],
                                  base_tag + tag_counter,
                                  comm,
                                  &request);
      DRAY_CHECK_MPI_ERROR(mpi_error);
      requests.push_back(request);
      tag_counter++;
    }
  }

  // recv it
  for(auto &domain : recv_buffers)
  {
    const int32 base_tag = domain.first;
    int32 tag_counter = 0;
    // TODO: check for max int size
    for(auto &buffer : domain.second)
    {
      MPI_Request request;
      int32 mpi_error = MPI_Irecv(buffer.second,
                                  static_cast<int>(buffer.first),
                                  MPI_BYTE,
                                  recv_srcs[base_tag],
                                  base_tag + tag_counter,
                                  comm,
                                  &request);
      DRAY_CHECK_MPI_ERROR(mpi_error);
      requests.push_back(request);
      tag_counter++;
    }
  }
  std::vector<MPI_Status> status;
  status.resize(requests.size());
  int32 mpi_error = MPI_Waitall(requests.size(), &requests[0], &status[0]);
  DRAY_CHECK_MPI_ERROR(mpi_error);

#if 0
  std::vector<conduit::relay::mpi::Request> sends;
  std::vector<conduit::relay::mpi::Request> recvs;
  std::vector<conduit::Node> send_nodes;
  std::vector<conduit::Node> recv_nodes;


  send_nodes.resize(send_count);
  sends.resize(send_count);
  recv_nodes.resize(recv_count);
  recvs.resize(recv_count);

  int32 send_idx = 0;
  int32 recv_idx = 0;
  for(int32 i = 0; i < total_comm; ++i)
  {
    CommInfo info = m_comm_info[i];
    bool send = info.m_src_rank == rank;
    const int32 base_tag = info.m_domain_id * 1000;

    if(send)
    {
      conduit::Node &n_domain = send_nodes[send_idx];
      DataSet domain = collection.domain(info.m_src_idx);
      domain.to_node(n_domain);
      detail::pack_dataset(n_domain, send_buffers[base_tag]);
      //conduit::relay::mpi::Request &send = sends[send_idx];
      conduit::relay::mpi::isend(n_domain,
                                 info.m_dest_rank,
                                 info.m_domain_id,
                                 comm,
                                 &sends[send_idx]);
      send_idx++;
      std::cout<<"send rank "<<rank<<" id "<<info.m_domain_id<<" size "<<n_domain.total_bytes_compact()<<"\n";
      //if(info.m_domain_id == 0) n_domain.schema().print();
    }
    else
    {

      DataSet &domain = m_recv_q[info.m_domain_id];
      conduit::Node &n_domain = recv_nodes[recv_idx];
      domain.to_node(n_domain);
      //conduit::relay::mpi::Request &recv = recvs[recv_idx];
      conduit::relay::mpi::irecv(n_domain,
                                 info.m_src_rank,
                                 info.m_domain_id,
                                 comm,
                                 &recvs[recv_idx]);

      std::cout<<"recv rank "<<rank<<" id "<<info.m_domain_id<<" size "<<n_domain.total_bytes_compact()<<"\n";
      //if(info.m_domain_id == 0) n_domain.schema().print();

      recv_idx++;
    }

  }

  if(send_count > 0)
  {
    std::vector<MPI_Status> status;
    status.resize(send_count);
    std::cout<<"rank "<<rank<<" waiting for sends "<<send_count<<"\n";
    conduit::relay::mpi::wait_all_send(send_count, &sends[0], &status[0]);
  }
  if(recv_count > 0)
  {
    std::vector<MPI_Status> status;
    status.resize(recv_count);
    std::cout<<"rank "<<rank<<" waiting for recvs "<<recv_count<<"\n";
    conduit::relay::mpi::wait_all_recv(recv_count, &recvs[0], &status[0]);
  }
#endif
  MPI_Barrier(comm);
#endif
}

}//namespace dray
