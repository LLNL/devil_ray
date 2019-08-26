#ifndef DRAY_HPP
#define DRAY_HPP

namespace dray
{

class dray
{
  public:
    void about();
    void init();
    void finalize();

    static void set_face_subdivisions(const int num_subdivions);
    static void set_zone_subdivisions(const int num_subdivions);

    static int get_face_subdivisions();
    static int get_zone_subdivisions();

  private:
    static int m_face_subdivisions;
    static int m_zone_subdivisions;
};

} // namespace dray
#endif
