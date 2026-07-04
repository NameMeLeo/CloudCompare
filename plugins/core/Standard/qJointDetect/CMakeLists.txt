# qJointDetect — ML-Powered Rock Joint Detection Plugin

set(PLUGIN_NAME "qJointDetect")
set(PLUGIN_DESC "ML-powered rock joint detection using Compass-labeled training data")
set(PLUGIN_ICON ":qJointDetect.png")

add_plugin(
    NAME ${PLUGIN_NAME}
    TYPE STANDARD
    CC_VERSION_MINIMAL "2.13"
    SOURCES
        src/ccJointDetect.cpp
        src/ccJointDetectDlg.cpp
        src/ccJointDetectTool.cpp
        src/JointFeatureExtractor.cpp
        src/JointMLClient.cpp
        src/JointOutputManager.cpp
    HEADERS
        include/ccJointDetect.h
        include/ccJointDetectDlg.h
        include/ccJointDetectTool.h
        include/JointFeatureExtractor.h
        include/JointMLClient.h
        include/JointOutputManager.h
    UI_FORMS
        ui/jointDetectDlg.ui
    RESOURCES
        qJointDetect.qrc
)

target_link_libraries(${PLUGIN_NAME} PRIVATE
    CCCoreLib
    Qt::Core
    Qt::Widgets
)

# ONNX Runtime (download/path configured by user or cmake preset)
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
    target_compile_definitions(${PLUGIN_NAME} PRIVATE HAS_ONNX_RUNTIME)
    target_link_libraries(${PLUGIN_NAME} PRIVATE onnxruntime::onnxruntime)
    message(STATUS "qJointDetect: ONNX Runtime found — ML inference enabled")
else()
    message(WARNING "qJointDetect: ONNX Runtime not found — inference disabled")
endif()

install_plugin(
    NAME ${PLUGIN_NAME}
)
