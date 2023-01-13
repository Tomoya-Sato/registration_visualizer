#ifndef OCCUPANCY_RATE_HPP
#define OCCUPANCY_RATE_HPP

#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct RateGrid
{
  int source_occupied = 0;
  int target_occupied = 0;
};

template <typename PointT>
void getMinMax2d(pcl::PointCloud<PointT> &source, double &min_x, double &max_x, double &min_y, double &max_y)
{
  min_x = std::numeric_limits<double>::max();
  max_x = std::numeric_limits<double>::min();
  min_y = std::numeric_limits<double>::max();
  max_y = std::numeric_limits<double>::min();

  for (auto p : source.points)
  {
    min_x = (min_x < p.x) ? min_x : p.x;
    max_x = (max_x > p.x) ? max_x : p.x;
    min_y = (min_y < p.y) ? min_y : p.y;
    max_y = (max_y > p.y) ? max_y : p.y;
  }
}

template <typename PointT>
double calcOccupancyRate(pcl::PointCloud<PointT> &source, pcl::PointCloud<PointT> &target, const double& grid_size)
{
  double min_x, max_x;
  double min_y, max_y;

  getMinMax2d(source, min_x, max_x, min_y, max_y);

  double min_x_bound = std::floor(min_x);
  double max_x_bound = std::ceil(max_x);
  double min_y_bound = std::floor(min_y);
  double max_y_bound = std::ceil(max_y);

  int x_grid_num = (max_x_bound - min_x_bound) / grid_size;
  int y_grid_num = (max_y_bound - min_y_bound) / grid_size;

  std::vector<RateGrid> grid_vec(x_grid_num * y_grid_num);
  
  for (auto p : source.points)
  {
    int x_index = std::floor((p.x - min_x_bound / grid_size));
    int y_index = std::floor((p.y - min_y_bound / grid_size));

    if (0 <= x_index && x_index < x_grid_num && 0 <= y_index && y_index < y_grid_num)
    {
      int xy_index = x_index + (x_grid_num * y_index);
      grid_vec[xy_index].source_occupied = 1;
    }
  }

  for (auto p : target.points)
  {
    int x_index = std::floor((p.x - min_x_bound / grid_size));
    int y_index = std::floor((p.y - min_y_bound / grid_size));

    if (0 <= x_index && x_index < x_grid_num && 0 <= y_index && y_index < y_grid_num)
    {
      int xy_index = x_index + (x_grid_num * y_index);
      grid_vec[xy_index].target_occupied = 1;
    }
  }

  int source_sum = 0, target_sum = 0;
  for (auto grid : grid_vec)
  {
    source_sum += grid.source_occupied;
    target_sum += (grid.source_occupied & grid.target_occupied);
  }

  double rate = static_cast<double>(target_sum) / static_cast<double>(source_sum);

  return rate;
}

#endif
