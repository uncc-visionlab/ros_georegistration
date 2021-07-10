/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//#include <ignition/math/Helpers.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/physics/physics.hh>
//#include <gazebo/common/CommonIface.hh>
#include <gazebo/transport/Node.hh>

#include "ModelSizePlugin.hh"

namespace gazebo {

    /// \brief Private data class for StaticMapPlugin

    class ModelSizePluginPrivate {
        /// \brief Download map tiles.
        /// \param[in] _centerLat Latitude of center point of map
        /// \param[in] _centerLon Longitude of center point of map
        /// \param[in] _zoom Map zoom level between 0 (entire world) and 21+
        /// (streets).
        /// \param[in] _tileSizePx Size of each map tile in pixels. Tiles will be
        /// square.
        /// \param[in] _worldSize Size of map in the world in meters.
        /// \param[in] _mapType Type of map to download: roadmap, satellite,
        /// terrain, hybrid
        /// \param[in] _apiKey Google API key
        /// \param[in] _saveDirPath Location in local filesystem to save tile
        /// images.
    public:
        //        std::vector<std::string> DownloadMapTiles(const double _centerLat,
        //                const double _centerLon, const unsigned int _zoom,
        //                const unsigned int _tileSizePx,
        //                const ignition::math::Vector2d &_worldSize,
        //                const std::string &_mapType, const std::string &_apiKey,
        //                const std::string &_saveDirPath);

        /// \brief Create textured map model and save it in specified path.
        /// \param[in] _name Name of map model
        /// \param[in] _tileWorldSize Size of map tiles in meters
        /// \param[in] _xNumTiles Number of tiles in x direction
        /// \param[in] _yNumTiles Number of tiles in y direction
        /// \param[in] _tiles Tile image filenames
        /// \param[in] _modelPath Path to model directory
        /// \return True if map tile model has been successfully created.
    public:
        //        bool CreateMapTileModel(
        //                const std::string &_name,
        //                const double _tileWorldSize,
        //                const unsigned int xNumTiles, const unsigned int yNumTiles,
        //                const std::vector<std::string> &_tiles, const std::string &_modelPath);

        /// \brief Get the ground resolution at the specified latitude and zoom
        /// level.
        /// \param[in] _lat Latitude
        /// \param[in] _zoom Map zoom Level
        /// \return Ground resolution in meters per pixel.
    public:
        //        double GroundResolution(const double _lat,
        //                const unsigned int _zoom) const;

        /// \brief Spawn a model into the world
        /// \param[in] _name Name of model
        /// \param[in] _pose Pose of model
    public:
        //        void SpawnModel(const std::string &_name,
        //                const ignition::math::Pose3d &_pose);

        /// \brief Pointer to world.
    public:
        physics::WorldPtr world;
        transport::NodePtr node;
        transport::PublisherPtr imagePub;
        transport::SubscriberPtr commandSubscriber;
        //        /// \brief Name of map model
        //    public:
        //        std::string modelName;
        //
        //        /// \brief Pose of map model
        //    public:
        //        ignition::math::Pose3d modelPose;
        //
        //        /// \brief Latitude and Longitude of map center
        //    public:
        //        ignition::math::Vector2d center;
        //
        //        /// \brief Target size of world to be covered by map in meters.
        //    public:
        //        ignition::math::Vector2d worldSize;
        //
        //        /// \brief Map zoom level. From 0 (entire world) to 21+ (streets)
        //    public:
        //        unsigned int zoom = 21u;
        //
        //        /// \brief Size of map tile in pixels. 640 is max resolution for users of
        //        /// standard API
        //    public:
        //        unsigned int tileSizePx = 640u;
        //
        //        /// \brief Type of map to use as texture: roadmap, satellite, terrain,
        //        /// hybrid
        //    public:
        //        std::string mapType = "satellite";
        //
        //        /// \brief True to use cached model and image data from gazebo model path.
        //        /// False to redownload image tiles and recreate model sdf and config
        //        /// files.
        //    public:
        //        bool useCache = false;
        //
        //        /// \brief Google API key
        //    public:
        //        std::string apiKey;
        //
        //        /// \brief Filenames of map tile images
        //    public:
        //        std::vector<std::string> mapTileFilenames;

        /// \brief Pointer to a node for communication.
    public:
        //transport::NodePtr node;

        /// \brief Factory publisher.
    public:
        //transport::PublisherPtr factoryPub;

        /// \brief True if the plugin is loaded successfully
    public:
        bool loaded = false;
    };
}

using namespace gazebo;

//const unsigned int MercatorProjection::TILE_SIZE = 256;

GZ_REGISTER_WORLD_PLUGIN(ModelSizePlugin)

/////////////////////////////////////////////////
ModelSizePlugin::ModelSizePlugin()
: dataPtr(new ModelSizePluginPrivate) {
}

/////////////////////////////////////////////////

void ModelSizePlugin::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) {
    this->dataPtr->world = _world;
    this->dataPtr->node = transport::NodePtr(new transport::Node());
    //world = _parent;
    // Initialize the node with the world name
#if GAZEBO_MAJOR_VERSION >= 8
    this->dataPtr->node->Init(this->dataPtr->world->Name());
#else
    this->dataPtr->node->Init(this->dataPtr->world->GetName());
#endif
    std::cout << "Subscribing to: " << "~/modelsize/request" << std::endl;
    this->dataPtr->commandSubscriber = this->dataPtr->node->Subscribe("~/modelsize/request",
            &ModelSizePlugin::getModelSize, this);
    std::cout << "Publishing to: " << "~/modelsize/response" << std::endl;
    this->dataPtr->imagePub = this->dataPtr->node->Advertise<model_bbox_msgs::msgs::ModelBoundingBoxResponse>("~/modelsize/response");
    //    if (!_sdf->HasElement("api_key")) {
    //        gzerr << "Missing Google API key needed to download map tiles" << std::endl;
    //        return;
    //    }

    //    if (!_sdf->HasElement("center")) {
    //        gzerr << "Please specify latitude and longitude coordinates of map center"
    //                << std::endl;
    //        return;
    //    }

    //    if (!_sdf->HasElement("world_size")) {
    //        gzerr << "Please specify size of map to cover in meters" << std::endl;
    //        return;
    //    }

    //    this->dataPtr->apiKey = _sdf->Get<std::string>("api_key");
    //    this->dataPtr->center =
    //            _sdf->Get<ignition::math::Vector2d>("center");
    //    double wSize = _sdf->Get<double>("world_size");
    //    if (wSize > 0) {
    //        // support only square map for now
    //        this->dataPtr->worldSize.X() = wSize;
    //        this->dataPtr->worldSize.Y() = wSize;
    //    } else {
    //        gzerr << "World size must be greater than 0 meters" << std::endl;
    //        return;
    //    }

    // optional params
    //    if (_sdf->HasElement("zoom"))
    //        this->dataPtr->zoom = _sdf->Get<unsigned int>("zoom");
    //
    //    if (_sdf->HasElement("tile_size")) {
    //        this->dataPtr->tileSizePx = _sdf->Get<unsigned int>("tile_size");
    //        if (this->dataPtr->tileSizePx > 640u) {
    //            gzerr << "Tile size exceeds standard API usage limit. Setting to 640px."
    //                    << std::endl;
    //            this->dataPtr->tileSizePx = 640u;
    //        }
    //    }

    //    if (_sdf->HasElement("map_type"))
    //        this->dataPtr->mapType = _sdf->Get<std::string>("map_type");
    //
    //    if (_sdf->HasElement("use_cache"))
    //        this->dataPtr->useCache = _sdf->Get<bool>("use_cache");
    //
    //    if (_sdf->HasElement("pose"))
    //        this->dataPtr->modelPose = _sdf->Get<ignition::math::Pose3d>("pose");
    //
    //    if (_sdf->HasElement("model_name"))
    //        this->dataPtr->modelName = _sdf->Get<std::string>("model_name");
    //    else {
    //        // generate name based on input
    //        std::stringstream name;
    //        name << "map_" << this->dataPtr->mapType << "_" << std::setprecision(9)
    //                << this->dataPtr->center.X() << "_" << this->dataPtr->center.Y()
    //                << "_" << this->dataPtr->worldSize.X()
    //                << "_" << this->dataPtr->worldSize.Y();
    //        this->dataPtr->modelName = name.str();
    //    }

    this->dataPtr->loaded = true;
}

/////////////////////////////////////////////////

void ModelSizePlugin::Init() {
    // don't init if params are not loaded successfully
    if (!this->dataPtr->loaded)
        return;

    // check if model exists locally
    //    auto basePath = common::SystemPaths::Instance()->GetLogPath() /
    //            boost::filesystem::path("models");

    //    this->dataPtr->node = transport::NodePtr(new transport::Node());
    //    this->dataPtr->node->Init();
    //    this->dataPtr->factoryPub =
    //            this->dataPtr->node->Advertise<msgs::Factory>("~/factory");

    //    boost::filesystem::path modelPath = basePath / this->dataPtr->modelName;
    //    if (this->dataPtr->useCache && common::exists(modelPath.string())) {
    //        gzmsg << "Model: '" << this->dataPtr->modelName << "' exists. "
    //                << "Spawning existing model.." << std::endl;
    //        this->dataPtr->SpawnModel("model://" + this->dataPtr->modelName,
    //                this->dataPtr->modelPose);
    //        return;
    //    }


    // create tmp dir to save model files
    //    boost::filesystem::path tmpModelPath =
    //            boost::filesystem::temp_directory_path() / this->dataPtr->modelName;
    //    boost::filesystem::path scriptsPath(tmpModelPath / "materials" / "scripts");
    //    boost::filesystem::create_directories(scriptsPath);
    //    boost::filesystem::path texturesPath(tmpModelPath / "materials" / "textures");
    //    boost::filesystem::create_directories(texturesPath);
    //
    //    // download map tile images into model/materials/textures
    //    std::vector<std::string> tiles = this->dataPtr->DownloadMapTiles(
    //            this->dataPtr->center.X(),
    //            this->dataPtr->center.Y(),
    //            this->dataPtr->zoom,
    //            this->dataPtr->tileSizePx,
    //            this->dataPtr->worldSize,
    //            this->dataPtr->mapType,
    //            this->dataPtr->apiKey,
    //            texturesPath.string());
    //
    //    // assume square model for now
    //    unsigned int xNumTiles = std::sqrt(tiles.size());
    //    unsigned int yNumTiles = xNumTiles;
    //
    //    double tileWorldSize = this->dataPtr->GroundResolution(
    //            IGN_DTOR(this->dataPtr->center.X()), this->dataPtr->zoom)
    //            * this->dataPtr->tileSizePx;
    //
    //    // create model and spawn it into the world
    //    if (this->dataPtr->CreateMapTileModel(
    //            this->dataPtr->modelName, tileWorldSize,
    //            xNumTiles, yNumTiles, tiles, tmpModelPath.string())) {
    //        // verify model dir is created
    //        if (common::exists(tmpModelPath.string())) {
    //            // remove existing map model
    //            if (common::exists(modelPath.string()))
    //                boost::filesystem::remove_all(modelPath);
    //
    //            try {
    //                // move new map model to gazebo model path
    //                boost::filesystem::rename(tmpModelPath, modelPath);
    //            } catch (boost::filesystem::filesystem_error &_e) {
    //                // rename failed. Could be an invalid cross-device link error
    //                // try copy and remove method
    //                bool result = common::copyDir(tmpModelPath, modelPath);
    //                if (result) {
    //                    boost::filesystem::remove_all(tmpModelPath);
    //                } else {
    //                    gzerr << "Unable to copy model from '" << tmpModelPath.string()
    //                            << "' to '" << modelPath.string() << "'" << std::endl;
    //                    return;
    //                }
    //            }
    //            // spawn the model
    //            this->dataPtr->SpawnModel("model://" + this->dataPtr->modelName,
    //                    this->dataPtr->modelPose);
    //        } else
    //            gzerr << "Failed to create model: " << tmpModelPath.string() << std::endl;
    //    }
}

void ModelSizePlugin::getModelSize(ModelBoundingBoxRequestPtr &req) {
    model_bbox_msgs::msgs::ModelBoundingBoxResponse res;
    gazebo::physics::ModelPtr model = this->dataPtr->world->ModelByName(req->model_name());
    std::cout << "Got Request: model_name: " <<  req->model_name() <<std::endl;
    if (!model) {
        std::cout << "GetModelProperties: model [%s] does not exist" << req->model_name() << std::endl;
        res.set_success(false);
        res.set_status_msg("GetModelProperties: model does not exist");
        return;
    } else {
        // get model parent name
        gazebo::physics::ModelPtr parent_model = boost::dynamic_pointer_cast<gazebo::physics::Model>(model->GetParent());

        // get list of child bodies, geoms
        for (unsigned int i = 0; i < model->GetChildCount(); i++) {
            gazebo::physics::LinkPtr body = boost::dynamic_pointer_cast<gazebo::physics::Link>(model->GetChild(i));
            if (body) {
                // get list of geoms
                for (unsigned int j = 0; j < body->GetChildCount(); j++) {
                    gazebo::physics::CollisionPtr geom = boost::dynamic_pointer_cast<gazebo::physics::Collision>(body->GetChild(j));

                    if (geom) {
                        res.set_geom_name( geom->GetName());
                        gazebo::physics::ShapePtr shape(geom->GetShape());
                        std::cout << "Found geometry " << std::endl;
                        if (shape->HasType(gazebo::physics::Base::BOX_SHAPE)) {
                            gazebo::physics::BoxShape *box = static_cast<gazebo::physics::BoxShape*> (shape.get());
                            ignition::math::Vector3d tmp_size = box->Size();
                            ignition::math::Vector3d tmp_scale = box->Scale();
                            *res.mutable_size() = gazebo::msgs::Convert(tmp_size);
                            *res.mutable_scale() = gazebo::msgs::Convert(tmp_scale);
                            std::cout << "Set BOX message data!" << std::endl;
                        } else if (shape->HasType(gazebo::physics::Base::PLANE_SHAPE)) {
                            gazebo::physics::PlaneShape *plane = static_cast<gazebo::physics::PlaneShape*> (shape.get());
                            ignition::math::Vector2d tmp_size2 = plane->Size();
                            //ignition::math::Vector2d tmp_scale2 = plane->Scale();
                            ignition::math::Vector3d tmp_size(tmp_size2[0], tmp_size2[1], 0);
                            ignition::math::Vector3d tmp_scale(0, 0, 0);
                            *res.mutable_size() = gazebo::msgs::Convert(tmp_size);
                            *res.mutable_scale() = gazebo::msgs::Convert(tmp_scale);
                            std::cout << "Set PLANE message data!" << std::endl;                            
                        }
                    }
                }
            }
        }
        res.set_success(true);
        res.set_status_msg("GetModelProperties: got properties");
        this->dataPtr->imagePub->Publish(res);
    }
}

