#include <dray/dray.hpp>
#include <iostream>

namespace dray
{

int dray::m_face_subdivisions = 1;
int dray::m_zone_subdivisions = 1;

void
dray::set_face_subdivisions(int num_subdivisions)
{
  m_face_subdivisions  = num_subdivisions;
}

void
dray::set_zone_subdivisions(int num_subdivisions)
{
  m_zone_subdivisions  = num_subdivisions;
}

int
dray::get_zone_subdivisions()
{
  return m_zone_subdivisions;
}

int
dray::get_face_subdivisions()
{
  return m_zone_subdivisions;
}

void
dray::init()
{

}

void
dray::finalize()
{

}

void
dray::about()
{
  std::cout<<"                                          v0.0.1                                      \n\n\n";
  std::cout<<"                                       @          &,                                      \n";
  std::cout<<"                                      @&          .@                                      \n";
  std::cout<<"                                      @%          .@*                                     \n";
  std::cout<<"                                    &@@@@@@@@@@@@@@@@@/                                   \n";
  std::cout<<"                                   @@@@@@@@@@@@@@@@@@@@.                                  \n";
  std::cout<<"                                   @@@@@@@@@@@@@@@@@@@@,                                  \n";
  std::cout<<"                                 /@@@@@@@@@@@@@@@@@@@@@@@                                 \n";
  std::cout<<"                               &@@@@@@@@@@@@@@@@@@@@@@@@@@@                               \n";
  std::cout<<"                         ,%&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*                         \n";
  std::cout<<"                   (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&                   \n";
  std::cout<<"              ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#              \n";
  std::cout<<"           (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.          \n";
  std::cout<<"        /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&        \n";
  std::cout<<"      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.     \n";
  std::cout<<"   .@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%   \n";
  std::cout<<" .@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@# \n";
  std::cout<<"@#                 /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&                 *@\n";
  std::cout<<"                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*                       \n";
  std::cout<<"                           ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                           \n";
  std::cout<<"                              &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                              \n";
  std::cout<<"                                 &@@@@@@@@@@@@@@@@@@@@@@@                                 \n";
  std::cout<<"                                    #@@@@@@@@@@@@@@@@@.                                   \n";
  std::cout<<"                                       .@@@@@@@@@@#                                       \n";
  std::cout<<"                                           @@@@.                                          \n";
  std::cout<<"                                            &@%                                           \n";
  std::cout<<"                                            ,@(                                           \n";
  std::cout<<"                                             @,                                           \n";
  std::cout<<"                                             @                                            \n";
  std::cout<<"                                           ,,@,,                                          \n";
  std::cout<<"                                      /@&.*  @  *,@@*                                     \n";
  std::cout<<"                                    %@       @       &&                                   \n";
  std::cout<<"                                   *&        @        %.                                  \n";
  std::cout<<"                                    @        @        @                                   \n";
  std::cout<<"                                     @       @       @.                                   \n";
  std::cout<<"                                      @      @      @.                                    \n";
  std::cout<<"                                      @      @      @                                     \n";
  std::cout<<"                                      @      @      @                                     \n";
  std::cout<<"                                   &@@       @       @@/                                  \n";
  std::cout<<"                                   @@.       @        @&                                  \n";
  std::cout<<"                                           (@@@&                                          \n";
  std::cout<<"                                            *@%                                           \n";
}

} // namespace dray
