#include "registration_visualizer.hpp"

#include <chrono>

#include <boost/bind.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <glk/pointcloud_buffer.hpp>
#include <glk/pointcloud_buffer_pcl.hpp>

#include <portable-file-dialogs.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>

#include "occupancy_rate.hpp"

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

  viewer->set_max_text_buffer_size(4);

  viewer->register_ui_callback("cloud_loader", [&]() {
    // if (ImGui::Button("Load Target Cloud"))
    // {
    //   std::vector<std::string> results = pfd::open_file("choose PCD file").result();
    //   if (!results.empty())
    //   {
    //     std::lock_guard<std::mutex> lock(mtx_);
    //     target_cloud_name_ = results[0];
    //     is_target_name_update_ = true;
    //   }
    // }
    // if (ImGui::Button("Load Source Cloud"))
    // {
    //   std::vector<std::string> results = pfd::open_file("choose PCD file").result();
    //   if (!results.empty())
    //   {
    //     std::lock_guard<std::mutex> lock(mtx_);
    //     source_cloud_name_ = results[0];
    //     is_source_name_update_ = true;
    //   }
    // }
    if (ImGui::Button("Load Cloud Set"))
    {
      // std::vector<std::string> results = pfd::open_file("choose PCD file", std::string("./"), {"All Files", "*"}, pfd::opt::force_path).result();
      std::string result = pfd::select_folder("Choose folder", std::string("./"), pfd::opt::force_path).result();
      if (!result.empty())
      {
        std::lock_guard<std::mutex> lock(mtx_);
        source_cloud_name_ = result + "/source_org.pcd";
        target_cloud_name_ = result + "/target.pcd";
        is_source_name_update_ = true;
        is_target_name_update_ = true;
      }
    }
  });

  viewer->register_ui_callback("Console", [&]() {
    ImGui::DragFloat("Target Leaf Size", &target_leaf_size_, 0.01f, 0.0f, 3.0f);
    ImGui::DragFloat("Source Leaf Size", &source_leaf_size_, 0.01f, 0.0f, 3.0f);
    ImGui::DragFloat("Epsilon", &epsilon_, 0.001f, 0.0f, 0.1f);
    ImGui::DragFloat("Resolution", &resolution_, 0.01f, 0.0f, 3.0f);

    if (ImGui::Button("Align"))
      do_align_ = true;

    if (ImGui::Button("Close"))
    {
      viewer->close();
      viewer_closed_ = true;
    }
  });

  viewer->register_ui_callback("method_switch", [&]{
    if (ImGui::Checkbox("use_gicp", &use_gicp_) && use_gicp_)
    {
      use_vgicp_ = false;
      use_ndt_ = false;
    }
    if (ImGui::Checkbox("use_vgicp", &use_vgicp_) && use_vgicp_)
    {
      use_gicp_ = false;
      use_ndt_ = false;
    }
    if (ImGui::Checkbox("use_ndt", &use_ndt_) && use_ndt_)
    {
      use_gicp_ = false;
      use_vgicp_ = false;
    }
  });

  bool draw_target = true;
  bool draw_source = true;
  bool draw_aligned = true;
  bool draw_filtered_target = true;
  bool draw_filtered_source = true;
  bool draw_filtered_aligned = true;
  viewer->register_ui_callback("rendering_switch", [&]{
    ImGui::Checkbox("target_cloud", &draw_target);
    ImGui::Checkbox("source_cloud", &draw_source);
    ImGui::Checkbox("aligned_cloud", &draw_aligned);
    ImGui::Checkbox("target_filtered_cloud", &draw_filtered_target);
    ImGui::Checkbox("source_filtered_cloud", &draw_filtered_source);
    ImGui::Checkbox("aligned_filtered_cloud", &draw_filtered_aligned);
  });

  viewer->register_drawable_filter("drawable_filter", [&](const std::string& drawable_name){
    if (!draw_target && drawable_name.find("target_cloud") != std::string::npos)
      return false;
    if (!draw_source && drawable_name.find("source_cloud") != std::string::npos)
      return false;
    if (!draw_aligned && drawable_name.find("aligned_cloud") != std::string::npos)
      return false;
    if (!draw_filtered_target && drawable_name.find("target_filtered_cloud") != std::string::npos)
      return false;
    if (!draw_filtered_source && drawable_name.find("source_filtered_cloud") != std::string::npos)
      return false;
    if (!draw_filtered_aligned && drawable_name.find("aligned_filtered_cloud") != std::string::npos)
      return false;

    return true;
  });

  while (viewer->spin_once() && !viewer_closed_)
  {
    checkTargetCloud(viewer);
    checkSourceCloud(viewer);
    checkAlignedCloud(viewer);
    checkFilteredTargetCloud(viewer);
    checkFilteredSourceCloud(viewer);
    checkFilteredAlignedCloud(viewer);
    checkScore(viewer);
  }
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

void RegistrationVisualizer::applyVGF(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, pcl::PointCloud<pcl::PointXYZ>& output, const double& leaf_size)
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
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_target_name_update_)
  {
    target_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(target_cloud_name_, *target_cloud_ptr_);
    is_target_cloud_update_ = true;
    is_target_name_update_ = false;
    is_target_set_ = true;
  }
}

void RegistrationVisualizer::checkSourceName()
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_source_name_update_)
  {
    source_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(source_cloud_name_, *source_cloud_ptr_);
    is_source_cloud_update_ = true;
    is_source_name_update_ = false;
    is_source_set_ = true;
  }
}

void RegistrationVisualizer::checkAlign()
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (do_align_ && is_target_set_ && is_source_set_)
  {
    std::cout << "Align" << std::endl;

    filtered_target_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(target_cloud_ptr_, *filtered_target_cloud_ptr_, target_leaf_size_);
    filtered_source_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(source_cloud_ptr_, *filtered_source_cloud_ptr_, source_leaf_size_);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    if (use_gicp_)
    {
      fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
      gicp.setMaxCorrespondenceDistance(1.0);
      gicp.setTransformationEpsilon(epsilon_);

      gicp.setInputTarget(filtered_target_cloud_ptr_);
      gicp.setInputSource(filtered_source_cloud_ptr_);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      gicp.align(dummy, guess_matrix_);

      aligned_matrix_ = gicp.getFinalTransformation();
      iteration_ = 0;
      score_ = gicp.getFitnessScore();
      converged_ = gicp.hasConverged();
    }
    else if (use_vgicp_)
    {
      fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> gicp;

      gicp.setResolution(resolution_);
      gicp.setTransformationEpsilon(epsilon_);

      gicp.setInputTarget(filtered_target_cloud_ptr_);
      gicp.setInputSource(filtered_source_cloud_ptr_);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      gicp.align(dummy, guess_matrix_);

      aligned_matrix_ = gicp.getFinalTransformation();
      iteration_ = 0;
      score_ = gicp.getFitnessScore();
      converged_ = gicp.hasConverged();
    }
    else if (use_ndt_)
    {
      pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

      ndt.setResolution(resolution_);
      ndt.setTransformationEpsilon(epsilon_);
      ndt.setStepSize(0.1);
      ndt.setMaximumIterations(30);

      ndt.setInputTarget(filtered_target_cloud_ptr_);
      ndt.setInputSource(filtered_source_cloud_ptr_);

      pcl::PointCloud<pcl::PointXYZ> dummy;
      ndt.align(dummy, guess_matrix_);

      aligned_matrix_ = ndt.getFinalTransformation();
      iteration_ = ndt.getFinalNumIteration();
      score_ = ndt.getTransformationProbability();
      converged_ = ndt.hasConverged();
    }

    end = std::chrono::system_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    time_ = duration;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_for_or(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_cloud_ptr_, *cloud_for_or, aligned_matrix_);
    occupancy_rate_ = calcOccupancyRate(*cloud_for_or, *target_cloud_ptr_, 0.5);

    // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    // ndt.setTransformationEpsilon(0.01);
    // ndt.setStepSize(0.1);
    // ndt.setResolution(2.0);
    // ndt.setMaximumIterations(100);

    // ndt.setInputTarget(filtered_target_cloud_ptr);
    // ndt.setInputSource(filtered_source_cloud_ptr);

    // pcl::io::savePCDFileBinary("test_target.pcd", *filtered_target_cloud_ptr);
    // pcl::io::savePCDFileBinary("test_source.pcd", *filtered_source_cloud_ptr);

    // pcl::PointCloud<pcl::PointXYZ> dummy;
    // ndt.align(dummy, guess_matrix_);

    // aligned_matrix_ = ndt.getFinalTransformation();
    // iteration_ = 0; // ndt.getFinalNumIteration();
    // score_ = 0; // ndt.getTransformationProbability();
    // converged_ = ndt.hasConverged();

    std::cout << aligned_matrix_ << std::endl;

    is_filtered_source_cloud_update_ = true;
    is_filtered_target_cloud_update_ = true;
    is_filtered_aligned_cloud_update_ = true;
    is_aligned_cloud_update_ = true;
    is_score_update_ = true;
    do_align_ = false;
  }
}

void RegistrationVisualizer::checkTargetCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_target_cloud_update_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = target_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = target_cloud_ptr_->points[i].getVector4fMap();
      // intensity_vec[i] = target_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    // cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("target_cloud", cloud_buffer, guik::FlatColor(1.0f, 1.0f, 1.0f, 1.0f).add("point_scale", 0.1f));

    is_target_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkSourceCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_source_cloud_update_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = source_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = source_cloud_ptr_->points[i].getVector4fMap();
      // intensity_vec[i] = source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    // cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    // viewer->update_drawable("source_cloud", cloud_buffer, guik::VertexColor(guess_matrix_).add("point_scale", 0.1f));
    viewer->update_drawable("source_cloud", cloud_buffer, guik::FlatColor(0.0f, 1.0f, 0.0f, 1.0f).add("point_scale", 0.1f));

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
      // intensity_vec[i] = source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    // cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    // viewer->update_drawable("aligned_cloud", cloud_buffer, guik::VertexColor(aligned_matrix_).add("point_scale", 0.1f));
    viewer->update_drawable("aligned_cloud", cloud_buffer, guik::FlatColor(1.0f, 0.0f, 0.0f, 0.0f, aligned_matrix_).add("point_scale", 0.1f));

    is_aligned_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkFilteredTargetCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_filtered_target_cloud_update_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = filtered_target_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = filtered_target_cloud_ptr_->points[i].getVector4fMap();
      // intensity_vec[i] = filtered_target_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    // cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("target_filtered_cloud", cloud_buffer, guik::FlatColor(1.0f, 1.0f, 1.0f, 1.0f).add("point_scale", 0.5f));

    is_filtered_target_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkFilteredSourceCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_filtered_source_cloud_update_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = filtered_source_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = filtered_source_cloud_ptr_->points[i].getVector4fMap();
      // intensity_vec[i] = filtered_source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    // cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    // viewer->update_drawable("source_cloud", cloud_buffer, guik::VertexColor(guess_matrix_).add("point_scale", 0.1f));
    viewer->update_drawable("source_filtered_cloud", cloud_buffer, guik::FlatColor(0.0f, 1.0f, 0.0f, 1.0f).add("point_scale", 0.5f));

    is_filtered_source_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkFilteredAlignedCloud(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_filtered_aligned_cloud_update_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int cloud_size = filtered_source_cloud_ptr_->size();
    xyz_cloud->resize(cloud_size);
    std::vector<float> intensity_vec(cloud_size);

    for (int i = 0; i < cloud_size; i++)
    {
      xyz_cloud->points[i].getVector4fMap() = filtered_source_cloud_ptr_->points[i].getVector4fMap();
      // intensity_vec[i] = filtered_source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    // cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    // viewer->update_drawable("aligned_cloud", cloud_buffer, guik::VertexColor(aligned_matrix_).add("point_scale", 0.1f));
    viewer->update_drawable("aligned_filtered_cloud", cloud_buffer, guik::FlatColor(1.0f, 0.0f, 0.0f, 0.0f, aligned_matrix_).add("point_scale", 0.5f));

    is_filtered_aligned_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkScore(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_score_update_)
  {
    std::string text = "NDT Score\n";
    text = text + "Duration : " + std::to_string(time_) + "\n";
    text = text + "Score    : " + std::to_string(score_) + "\n";
    text = text + "OR       : " + std::to_string(occupancy_rate_);
    viewer->append_text(text);
    is_score_update_ = false;
  }
}
}
