#ifndef REGISTRATION_VISUALIZER_HPP
#define REGISTRATION_VISUALIZER_HPP

#include <mutex>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <guik/viewer/light_viewer.hpp>

namespace visualizer
{
class RegistrationVisualizer
{
public:
  RegistrationVisualizer();
  ~RegistrationVisualizer();
  void run();

private:
  std::string source_cloud_name_, target_cloud_name_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud_ptr_, target_cloud_ptr_;
  Eigen::Matrix4f guess_matrix_, aligned_matrix_;

  // Update flag
  bool is_source_name_update_ = false;
  bool is_target_name_update_ = false;
  bool is_source_cloud_update_ = false;
  bool is_target_cloud_update_ = false;
  bool is_aligned_cloud_update_ = false;
  bool is_guess_matrix_update_ = false;
  bool is_score_update_ = false;
  bool do_align_ = false;

  bool is_source_set_ = false;
  bool is_target_set_ = false;

  // Mutex
  std::mutex mtx_;

  // Thread
  std::thread viewer_thread_;

  bool viewer_closed_ = true;

  // Visualize
  float intensity_range_ = 7000.0f;
  float target_leaf_size_ = 0.1;
  float source_leaf_size_ = 1.0;
  int iteration_ = 0;
  double score_ = 0.0;
  bool converged_ = false;

  void applyVGF(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, pcl::PointCloud<pcl::PointXYZ>& output, const double& leaf_size);

  void viewerLoop();
  // Parent
  void checkTargetName();
  void checkSourceName();
  void checkAlign();

  // Child
  void checkTargetCloud(const std::shared_ptr<guik::LightViewer>& viewer);
  void checkSourceCloud(const std::shared_ptr<guik::LightViewer>& viewer);
  void checkAlignedCloud(const std::shared_ptr<guik::LightViewer>& viewer);
  void checkScore(const std::shared_ptr<guik::LightViewer>& viewer);
};
}

#endif
