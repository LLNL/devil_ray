#include <dray/filters/redistribute.hpp>
#include <dray/dray_node_to_dataset.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>

#include <conduit.hpp>
#include <algorithm>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#endif

namespace dray
{

namespace detail
{

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
      if(rank == 0) n_domain_meta.print();
      // allocate a data set to for recvs
      m_recv_q[info.m_domain_id] = to_dataset(n_domain_meta);
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

  std::vector<conduit::Node> send_nodes;
  std::vector<conduit::Node> recv_nodes;
  std::vector<conduit::relay::mpi::Request> sends;
  std::vector<conduit::relay::mpi::Request> recvs;

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

    if(send)
    {
      conduit::Node &n_domain = send_nodes[send_idx];
      DataSet domain = collection.domain(info.m_src_idx);
      domain.to_node(n_domain);

      conduit::relay::mpi::Request &send = sends[send_idx];
      conduit::relay::mpi::isend(n_domain,
                                 info.m_dest_rank,
                                 info.m_domain_id,
                                 comm,
                                 &send);
      send_idx++;
    }
    else
    {

      DataSet &domain = m_recv_q[info.m_domain_id];
      conduit::Node &n_domain = recv_nodes[recv_idx];
      domain.to_node(n_domain);
      conduit::relay::mpi::Request &recv = recvs[recv_idx];
      conduit::relay::mpi::irecv(n_domain,
                                 info.m_src_rank,
                                 info.m_domain_id,
                                 comm,
                                 &recv);


      recv_idx++;
    }

    if(send_count > 0)
    {
      std::vector<MPI_Status> status;
      status.resize(send_count);
      conduit::relay::mpi::wait_all_send(send_count, &sends[0], &status[0]);
    }
    if(recv_count > 0)
    {
      std::vector<MPI_Status> status;
      status.resize(recv_count);
      conduit::relay::mpi::wait_all_send(recv_count, &recvs[0], &status[0]);
    }
  }
#endif
}

}//namespace dray
