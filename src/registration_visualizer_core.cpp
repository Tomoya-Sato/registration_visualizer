#include "registration_visualizer.hpp"

#include <boost/bind.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <glk/pointcloud_buffer.hpp>
#include <glk/pointcloud_buffer_pcl.hpp>

#include <portable-file-dialogs.h>

#include <fast_gicp/gicp/fast_gicp.hpp>

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

  viewer->register_ui_callback("Console", [&]() {
    ImGui::DragFloat("Target Leaf Size", &target_leaf_size_, 0.01f, 0.0f, 3.0f);
    ImGui::DragFloat("Source Leaf Size", &source_leaf_size_, 0.01f, 0.0f, 3.0f);

    if (ImGui::Button("Align"))
      do_align_ = true;

    if (ImGui::Button("Close"))
    {
      viewer->close();
      viewer_closed_ = true;
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

  while (viewer->spin_once() && !viewer_closed_)
  {
    checkTargetCloud(viewer);
    checkSourceCloud(viewer);
    checkAlignedCloud(viewer);
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
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_target_name_update_)
  {
    target_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
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
    source_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(target_cloud_ptr_, *filtered_target_cloud_ptr, target_leaf_size_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    applyVGF(source_cloud_ptr_, *filtered_source_cloud_ptr, source_leaf_size_);

    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setMaxCorrespondenceDistance(1.0);
    gicp.setInputTarget(filtered_target_cloud_ptr);
    gicp.setInputSource(filtered_source_cloud_ptr);

    pcl::PointCloud<pcl::PointXYZ> dummy;
    gicp.align(dummy, guess_matrix_);

    aligned_matrix_ = gicp.getFinalTransformation();
    iteration_ = 0;
    score_ = gicp.getFitnessScore();
    converged_ = gicp.hasConverged();

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
      intensity_vec[i] = target_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("target_cloud", cloud_buffer, guik::VertexColor().add("point_scale", 0.1f));

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
      intensity_vec[i] = source_cloud_ptr_->points[i].intensity;
    }

    auto cloud_buffer = glk::create_point_cloud_buffer(*xyz_cloud);
    cloud_buffer->add_color(getColorVec(intensity_vec, intensity_range_, 1.0f)[0].data(), sizeof(Eigen::Vector4f), cloud_size);
    viewer->update_drawable("source_cloud", cloud_buffer, guik::VertexColor(guess_matrix_).add("point_scale", 0.1f));

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
    viewer->update_drawable("aligned_cloud", cloud_buffer, guik::VertexColor(aligned_matrix_).add("point_scale", 0.1f));

    is_aligned_cloud_update_ = false;
  }
}

void RegistrationVisualizer::checkScore(const std::shared_ptr<guik::LightViewer>& viewer)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (is_score_update_)
  {
    std::string text = "NDT Score\n";
    text = text + "Iteration: " + std::to_string(iteration_) + "\n";
    text = text + "Score    : " + std::to_string(score_) + "\n";
    text = text + "Converged: " + std::to_string(converged_);
    viewer->append_text(text);
    is_score_update_ = false;
  }
}
}
