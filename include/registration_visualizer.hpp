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
enum class METHOD
{
  FastGICP = 0,
  FastVGICP = 1,
  FastVGICPCuda = 2,
  PCL_NDT= 3,
};

class RegistrationVisualizer
{
public:
  RegistrationVisualizer();
  ~RegistrationVisualizer();
  void run();

private:
  std::string source_cloud_name_, target_cloud_name_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud_ptr_, target_cloud_ptr_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_master_cloud_ptr_, target_master_cloud_ptr_;
  Eigen::Matrix4f guess_matrix_, aligned_matrix_;

  // Update flag
  bool is_source_name_update_ = false;
  bool is_target_name_update_ = false;
  bool is_source_cloud_update_ = false;
  bool is_target_cloud_update_ = false;
  bool is_aligned_cloud_update_ = false;
  bool is_guess_matrix_update_ = false;
  bool is_score_update_ = false;
  bool is_intensity_thresh_update_ = false;
  bool do_align_ = false;
  bool read_next_ = false;

  bool is_source_set_ = false;
  bool is_target_set_ = false;

  // Method
  METHOD method_;
  bool use_gicp_ = true;
  bool use_vgicp_ = false;
  bool use_vgicp_cuda_ = false;
  bool use_pcl_ndt_ = false;

  // Params
  float resolution_ = 1.0;
  float epsilon_ = 0.01;
  float target_leaf_size_ = 0.1;
  float source_leaf_size_ = 1.0;
  float correspondence_distance_ = 1.0;
  float intensity_thresh_ = 0.0;
  float z_thresh_ = 10;

  // Mutex
  std::mutex mtx_;

  // Thread
  std::thread viewer_thread_;

  bool viewer_closed_ = true;

  std::vector<std::string> directories_;
  int dir_idx_ = -1;

  // Visualize
  float intensity_range_ = 60.0f;
  float scale_ = 1.0f;
  int iteration_ = 0;
  double score_ = 0.0;
  bool converged_ = false;
  double time_;

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
