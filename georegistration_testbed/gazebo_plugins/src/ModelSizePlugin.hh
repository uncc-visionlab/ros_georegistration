/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ModelSizePlugin.hh
 * Author: arwillis
 *
 * Created on August 3, 2020, 8:53 AM
 */

#ifndef MODELSIZEPLUGIN_HH
#define MODELSIZEPLUGIN_HH

#include <string>
#include <memory>

#include <gazebo/common/Plugin.hh>
#include <gazebo/util/system.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo_msgs/LinkState.h>

#include "model_bbox_request.pb.h"
#include "model_bbox_response.pb.h"
//#include "vector2d.pb.h"
#include "vector3d.pb.h"

namespace gazebo {

    typedef const boost::shared_ptr<const model_bbox_msgs::msgs::ModelBoundingBoxRequest>
    ModelBoundingBoxRequestPtr;

    typedef const boost::shared_ptr<const model_bbox_msgs::msgs::ModelBoundingBoxResponse>
    ModelBoundingBoxResponsePtr;

    /// \brief Forward declarations
    class ModelSizePluginPrivate;

    class GZ_PLUGIN_VISIBLE ModelSizePlugin : public WorldPlugin {
        /// \brief Constructor.
    public:
        ModelSizePlugin();

        /// \brief Load the plugin.
        /// \param[in] _world Pointer to world
        /// \param[in] _sdf Pointer to the SDF configuration.
    public:
        virtual void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf);

        /// \brief Initialize the plugin.
    public:
        virtual void Init();

    public:
        void getModelSize(ModelBoundingBoxRequestPtr &req);
        /// \brief Pointer to private data.
    private:
        std::unique_ptr<ModelSizePluginPrivate> dataPtr;
    };
}

#endif /* MODELSIZEPLUGIN_HH */

