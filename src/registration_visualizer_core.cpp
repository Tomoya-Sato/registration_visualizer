#include "registration_visualizer.hpp"

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <glk/pointcloud_buffer.hpp>
#include <glk/pointcloud_buffer_pcl.hpp>
#include <guik/model_control.hpp>

#include <portable-file-dialogs.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

namespace visualizer
{
std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> getColorVec(const std::vector<float>& intensity_vec, const float& intensity_range, const float& alpha)
{
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> color_vec(intensity_vec.size());
  for (int i = 0; i < intensity_vec.size(); i++)
  {
    float g = intensity_vec[i] / intensity_range;
    color_vec[i] = Eigen::Vector4f(1.0f, g, 0.0f, alpha);
  }
  return color_vec;
}

RegistrationVisualizer::RegistrationVisualizer()
{
  for (const auto& entry : boost::filesystem::directory_iterator(std::string("/home/tomoya/sandbox/map4_engine/debug/")))
  {
    if (boost::filesystem::is_directory(entry.path()))
      directories_.push_back(entry.path().string());
  }
  std::sort(directories_.begin(), directories_.end());

  guess_matrix_ = Eigen::Matrix4f::Identity();
  viewer_closed_ = false;
  viewer_thread_ = std::thread(boost::bind(&RegistrationVisualizer::viewerLoop, this));
}

RegistrationVisualizer::~RegistrationVisualizer()
{
}

void RegistrationVisualizer::viewerLoop()
{
  auto viewer = guik::LightViewer::instance();

  viewer->set_max_text_buffer_size(5);
  viewer->use_arcball_camera_control();

  viewer->register_ui_callback("1-cloud_loader", [&]() {
    if (ImGui::Button("Load Target Cloud"))
    {
      std::vector<std::string> results = pfd::open_file("choose PCD file").result();
      if (!results.empty())
      {
        std::lock_guard<std::mutex> lock(mtx_);
        target_cloud_name_ = results[0];
        is_target_name_update_ = true;
      }
    }
    if (ImGui::Button("Load Source Cloud"))
    {
      std::vector<std::string> results = pfd::open_file("choose PCD file").result();
      if (!results.empty())
      {
        std::lock_guard<std::mutex> lock(mtx_);
        source_cloud_name_ = results[0];
        is_source_name_update_ = true;
      }
    }
  });

  viewer->register_ui_callback("2-Console", [&]() {
    ImGui::DragFloat("Target Leaf Size", &target_leaf_size_, 0.01f, 0.0f, 3.0f);
    ImGui::DragFloat("Source Leaf Size", &source_leaf_size_, 0.01f, 0.0f, 3.0f);
    ImGui::DragFloat("Resolution", &resolution_, 0.01f, 0.0, 3.0f);
    ImGui::DragFloat("Correspondence Distance", &correspondence_distance_, 0.1f, 0.0, 10.0f);
    if (ImGui::DragFloat("Intensity Threshold", &intensity_thresh_, 1.0f, 0.0f, 255.0f))
    {
      is_source_cloud_update_ = true;
      is_target_cloud_update_ = true;
    }
    if (ImGui::DragFloat("Height Threshold", &z_thresh_, 0.1f, 0.0f, 10.0f))
    {
      is_source_cloud_update_ = true;
      is_target_cloud_update_ = true;
    }

    if (ImGui::Checkbox("FastGICP", &use_gicp_) && use_gicp_)
      use_vgicp_ = use_vgicp_cuda_ = use_pcl_ndt_ = false;

    if (ImGui::Checkbox("FastVGICP", &use_vgicp_) && use_vgicp_)
      use_gicp_ = use_vgicp_cuda_ = use_pcl_ndt_ = false;

    if (ImGui::Checkbox("FastVGICPCuda", &use_vgicp_cuda_) && use_vgicp_cuda_)
      use_gicp_ = use_vgicp_ = use_pcl_ndt_ = false;

    if (ImGui::Checkbox("PCL_NDT", &use_pcl_ndt_) && use_pcl_ndt_)
      use_gicp_ = use_vgicp_ = use_vgicp_cuda_ = false;

    if (ImGui::Button("Align"))
      do_align_ = true;

    if (ImGui::Button("Close"))
      viewer_closed_ = true;

    if (ImGui::Button("Next"))
    {
      dir_idx_++;
      std::lock_guard<std::mutex> lock(mtx_);
      target_cloud_name_ = directories_[dir_idx_] + "/target.pcd";
      source_cloud_name_ = directories_[dir_idx_] + "/source_ali.pcd";
      is_target_name_update_ = true;
      is_source_name_update_ = true;
      viewer->append_text(directories_[dir_idx_]);
    }
  });

  bool draw_target = true;
  bool draw_source = true;
  bool draw_aligned = true;
  viewer->register_ui_callback("rendering_switch", [&]{
    ImGui::Checkbox("target_cloud", &draw_target);
    ImGui::Checkbox("source_cloud", &draw_source);
    ImGui::Checkbox("aligned_cloud", &draw_aligned);
  });

  viewer->register_drawable_filter("drawable_filter", [&](const std::string& drawable_name){
    if (!draw_target && drawable_name.find("target_cloud") != std::string::npos)
      return false;
    if (!draw_source && drawable_name.find("source_cloud") != std::string::npos)
      return false;
    if (!draw_aligned && drawable_name.find("aligned_cloud") != std::string::npos)
      return false;

    return true;
  });

  // Guizmo
  Eigen::Matrix4f init_model_matrix = Eigen::Matrix4f::Identity();
  guik::ModelControl model_control("model_control", init_model_matrix);

  viewer->register_ui_callback("model_control_ui", [&]{
    model_control.draw_gizmo_ui();
    model_control.draw_gizmo();

    guess_matrix_ = model_control.model_matrix();
    std::lock_guard<std::mutex> lock(mtx_);
    is_source_cloud_update_ = true;
  });

  while (viewer->spin_once() && !viewer_closed_)
  {
    checkTargetCloud(viewer);
    checkSourceCloud(viewer);
    checkAlignedCloud(viewer);
    checkScore(viewer);
  }

  viewer->destroy();
}

void RegistrationVisualizer::run()
{
  while (!viewer_closed_)
  {
    checkTargetName();
    checkSourceName();
    checkAlign();
  }

  viewer_thread_.join();
}

void RegistrationVisualizer::applyVGF(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, pcl::PointCloud<pcl::PointXYZ>& output, const double& leaf_size)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  xyz_cloud->reserve(input->size());
  for (auto p : input->points)
  {
    pcl::PointXYZ q;
    q.getVector4fMap() = p.getVector4fMap();
    xyz_cloud->push_back(q);
  }

  if (leaf_size <= 0.0)
  {
    output = *xyz_cloud;
  }
  else
  {
    pcl::VoxelGrid<pcl::PointXYZ> vgf;
    vgf.setLeafSize(leaf_size, leaf_size, leaf_size);
    vgf.setInputCloud(xyz_cloud);
    vgf.filter(output);
  }
}

void RegistrationVisualizer::checkTargetName()
{
  if (is_target_name_update_)
  {
    target_master_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(target_cloud_name_, *target_master_cloud_ptr_);

    // Intensity normalization
    float max_intensity = 0;
    for (const auto& p : *target_master_cloud_ptr_)
      max_intensity = (p.intensity > max_intensity) ? p.intensity : max_intensity;

    for (auto& p : *target_master_cloud_ptr_)
      p.intensity = p.intensity / max_intensity * 255.0;

    std::lock_guard<std::mutex> lock(mtx_);
    is_target_cloud_update_ = true;
    is_target_name_update_ = false;
    is_target_set_ = true;
  }
}

void RegistrationVisualizer::checkSourceName()
{
  if (is_source_name_update_)
  {
    source_master_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(source_cloud_name_, *source_master_cloud_ptr_);

    // Intensity normalization
    float max_intensity = 0;
    for (const auto& p : *source_master_cloud_ptr_)
      max_intensity = (p.intensity > max_intensity) ? p.intensity : max_intensity;

    for (auto& p : *source_master_cloud_ptr_)
      p.intensity = p.intensity / max_intensity * 255.0;

    std::lock_guard<std::mutex> lock(mtx_);
    is_source_cloud_update_ = true;
    is_source_name_update_ = false;
    is_source_set_ = true;
  }
}

void RegistrationVisualizer::checkAlign()
{
  if (do_align_ && is_target_set_ && is_source_set_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_master_target_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(target_master_cloud_ptr_, *filtered_master_target_cloud_ptr, target_leaf_size_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_master_source_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(source_master_cloud_ptr_, *filtered_master_source_cloud_ptr, source_leaf_size_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(target_cloud_ptr_, *filtered_target_cloud_ptr, target_leaf_size_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(source_cloud_ptr_, *filtered_source_cloud_ptr, source_leaf_size_);

    std::chrono::time_point<std::chrono::system_clock> c_start, c_end;
    c_start = std::chrono::system_clock::now();

    // if (method_ == METHOD::FastGICP)
    if (use_gicp_)
    {
      fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
      gicp.setMaxCorrespondenceDistance(correspondence_distance_);

      // gicp.setInputTarget(filtered_master_target_cloud_ptr);
      // gicp.setInputSource(filtered_master_source_cloud_ptr);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      // gicp.align(dummy, guess_matrix_);

      // aligned_matrix_ = gicp.getFinalTransformation();

      gicp.setInputTarget(filtered_target_cloud_ptr);
      gicp.setInputSource(filtered_source_cloud_ptr);
      gicp.setMaximumIterations(100);

      gicp.align(dummy, guess_matrix_);

      aligned_matrix_ = gicp.getFinalTransformation();
      iteration_ = -1;
      score_ = gicp.getFitnessScore();
      converged_ = gicp.hasConverged();
    }
    // else if (method_ == METHOD::FastVGICP)
    else if (use_vgicp_)
    {
      fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
      vgicp.setResolution(resolution_);

      vgicp.setInputTarget(filtered_target_cloud_ptr);
      vgicp.setInputSource(filtered_source_cloud_ptr);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      vgicp.align(dummy, guess_matrix_);

      aligned_matrix_ = vgicp.getFinalTransformation();
      iteration_ = -1;
      score_ = vgicp.getFitnessScore();
      converged_ = vgicp.hasConverged();
    }
#ifdef USE_VGICP_CUDA
    // else if (method_ == METHOD::FastVGICPCuda)
    else if (use_vgicp_cuda_)
    {
      fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp;

      vgicp.setResolution(resolution_);
      vgicp.setTransformationEpsilon(epsilon_);

      vgicp.setInputTarget(filtered_target_cloud_ptr);
      vgicp.setInputSource(filtered_source_cloud_ptr);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      vgicp.align(dummy, guess_matrix_);

      aligned_matrix_ = vgicp.getFinalTransformation();
      iteration_ = 0;
      score_ = vgicp.getFitnessScore();
      converged_ = vgicp.hasConverged();
    }
#endif
    else // if (use_pcl_ndt_)
    {
      pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

      ndt.setResolution(resolution_);
      ndt.setTransformationEpsilon(epsilon_);
      ndt.setStepSize(0.1);
      ndt.setMaximumIterations(30);

      ndt.setInputTarget(filtered_target_cloud_ptr);
      ndt.setInputSource(filtered_source_cloud_ptr);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      ndt.align(dummy, guess_matrix_);

      aligned_matrix_ = ndt.getFinalTransformation();
      iteration_ = ndt.getFinalNumIteration();
      score_ = ndt.getTransformationProbability();
      converged_ = ndt.hasConverged();
    }

    c_end = std::chrono::system_clock::now();
    time_ = std::chrono::duration_cast<std::chrono::microseconds>(c_end - c_start).count() / 1000.0;

    std::lock_guard<std::mutex> lock(mtx_);
    is_aligned_cloud_update_ = true;
    is_score_update_ = true;
    do_align_ = false;
  }
}

void RegistrationVisualizer::checkTargetCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  if (is_target_cloud_update_ && is_target_set_)
  {
    target_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& p : *target_master_cloud_ptr_)
    {
      if (p.intensity >= intensity_thresh_ && p.z <= z_thresh_)
        target_cloud_ptr_->push_back(p);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = target_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = target_cloud_ptr_->points[i].getVector4fMap();
      intensity_vec[i] = target_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("target_cloud", cloud_buffer, guik::FlatColor(1.0f, 1.0f, 1.0f, 1.0f).add("point_scale", scale_));

    std::lock_guard<std::mutex> lock(mtx_);
    is_target_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkSourceCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  if (is_source_cloud_update_ && is_source_set_)
  {
    source_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& p : *source_master_cloud_ptr_)
    {
      if (p.intensity >= intensity_thresh_ && p.z <= z_thresh_)
        source_cloud_ptr_->push_back(p);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = source_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = source_cloud_ptr_->points[i].getVector4fMap();
      intensity_vec[i] = source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("source_cloud", cloud_buffer, guik::FlatColor(0.0f, 1.0f, 0.0f, 1.0f, guess_matrix_).add("point_scale", scale_));

    std::lock_guard<std::mutex> lock(mtx_);
    is_source_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkAlignedCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_aligned_cloud_update_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = source_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = source_cloud_ptr_->points[i].getVector4fMap();
      intensity_vec[i] = source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("aligned_cloud", cloud_buffer, guik::FlatColor(1.0f, 0.0f, 0.0f, 1.0f, aligned_matrix_).add("point_scale", scale_));

    is_aligned_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkScore(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_score_update_)
  {
    std::string text = "Registration Score\n";
    text = text + "Iteration: " + std::to_string(iteration_) + "\n";
    text = text + "Score    : " + std::to_string(score_) + "\n";
    text = text + "Converged: " + std::to_string(converged_) + "\n";
    text = text + "Duration : " + std::to_string(time_);
    viewer->append_text(text);
    is_score_update_ = false;
  }
}
}
