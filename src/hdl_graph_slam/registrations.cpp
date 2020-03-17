#include <hdl_graph_slam/registrations.hpp>

#include <iostream>

#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

#ifdef USE_FASTGICP
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif
#endif

namespace hdl_graph_slam {

boost::shared_ptr<pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>> select_registration_method(ros::NodeHandle& pnh) {
  using PointT = pcl::PointXYZI;

  // select a registration method (ICP, GICP, NDT)
  std::string registration_method = pnh.param<std::string>("registration_method", "NDT_OMP");
  if(registration_method == "ICP") {
    std::cout << "registration: ICP" << std::endl;
    boost::shared_ptr<pcl::IterativeClosestPoint<PointT, PointT>> icp(new pcl::IterativeClosestPoint<PointT, PointT>());
    icp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
    icp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
    icp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
    return icp;
  } else if(registration_method.find("GICP") != std::string::npos) {
    if(registration_method.find("OMP") == std::string::npos) {
      std::cout << "registration: GICP" << std::endl;
      boost::shared_ptr<pcl::GeneralizedIterativeClosestPoint<PointT, PointT>> gicp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
      gicp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
      gicp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
      gicp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
      gicp->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
      gicp->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));
      return gicp;
    } else {
      std::cout << "registration: GICP_OMP" << std::endl;
      boost::shared_ptr<pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>> gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>());
      gicp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
      gicp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
      gicp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
      gicp->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
      gicp->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));
      return gicp;
    }
  } else if (registration_method.find("NDT") != std::string::npos) {

    double ndt_resolution = pnh.param<double>("ndt_resolution", 0.5);
    if(registration_method.find("OMP") == std::string::npos) {
      std::cout << "registration: NDT " << ndt_resolution << std::endl;
      boost::shared_ptr<pcl::NormalDistributionsTransform<PointT, PointT>> ndt(new pcl::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
      ndt->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
      ndt->setResolution(ndt_resolution);
      return ndt;
    } else {
      int num_threads = pnh.param<int>("ndt_num_threads", 0);
      std::string nn_search_method = pnh.param<std::string>("ndt_nn_search_method", "DIRECT7");
      std::cout << "registration: NDT_OMP " << nn_search_method << " " << ndt_resolution << " (" << num_threads << " threads)" << std::endl;
      boost::shared_ptr<pclomp::NormalDistributionsTransform<PointT, PointT>> ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      if(num_threads > 0) {
        ndt->setNumThreads(num_threads);
      }
      ndt->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
      ndt->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
      ndt->setResolution(ndt_resolution);
      if(nn_search_method == "KDTREE") {
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      } else if (nn_search_method == "DIRECT1") {
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else {
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      }
      return ndt;
    }
  }
  #ifdef USE_FASTGICP
   else {
    if(registration_method.find("VGICP") != std::string::npos ) {
      std::cerr << "warning: unknown registration type(" << registration_method << ")" << std::endl;
      std::cerr << "       : use FGICP" << std::endl;
    }

    double vgicp_resolution = pnh.param<double>("vgicp_resolution", 1.0);

    if(registration_method.find("CUDA") == std::string::npos) {
      std::cout << "registration: VGICP_MT" << std::endl;
      boost::shared_ptr<fast_gicp::FastVGICP<PointT, PointT>> fast_vgicp(new fast_gicp::FastVGICP<PointT, PointT>());
      fast_vgicp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
      fast_vgicp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
      //fast_vgicp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
      fast_vgicp->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
      //fast_vgicp->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));

      fast_vgicp->setResolution(vgicp_resolution);

      return fast_vgicp;
    } else {
      #ifdef USE_VGICP_CUDA
      boost::shared_ptr<fast_gicp::FastVGICPCuda<PointT, PointT>> fast_vgicp_cuda(new fast_gicp::FastVGICPCuda<PointT, PointT>());

      fast_vgicp_cuda->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
      fast_vgicp_cuda->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
      //fast_vgicp_cuda->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
      fast_vgicp_cuda->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
      //fast_vgicp_cuda->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));

      fast_vgicp_cuda->setResolution(vgicp_resolution);


      // use GPU-based bruteforce nearest neighbor search for covariance estimation
      // this would be a good choice if your PC has a weak CPU and a strong GPU (e.g., NVIDIA Jetson)
      if (pnh.param<bool>("gpu_bruteforce", false))
        fast_vgicp_cuda->setNearesetNeighborSearchMethod(fast_gicp::GPU_BRUTEFORCE);
      
      // vgicp_cuda uses CPU-based parallel KDTree in covariance estimation by default
      // on a modern CPU, it is faster than GPU_BRUTEFORCE
      // vgicp_cuda.setNearesetNeighborSearchMethod(fast_gicp::CPU_PARALLEL_KDTREE);
      else
        fast_vgicp_cuda->setNearesetNeighborSearchMethod(fast_gicp::CPU_PARALLEL_KDTREE);

      return fast_vgicp_cuda;
      #endif
    }
  }
  #endif

  return nullptr;
}

}
