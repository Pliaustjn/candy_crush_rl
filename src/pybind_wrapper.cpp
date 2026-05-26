#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "candy_crush.h"

namespace py = pybind11;

PYBIND11_MODULE(candy_crush_cpp, m) {
    m.doc() = "Candy Crush Saga Environment Simulator (C++ backend)";

    py::class_<CandyCrushEnv>(m, "CandyCrushEnv")
        .def(py::init<int, int, int>(),
             py::arg("max_steps") = 100,
             py::arg("target_score") = 1000,
             py::arg("seed") = -1)
        .def("reset", &CandyCrushEnv::reset)
        .def("step", &CandyCrushEnv::step)
        .def("set_state", &CandyCrushEnv::set_state,
             py::arg("new_board"),
             py::arg("new_steps"),
             py::arg("new_score"))
        .def("clone", &CandyCrushEnv::clone)
        .def("get_valid_actions", &CandyCrushEnv::get_valid_actions)
        .def("render", &CandyCrushEnv::render)
        .def("get_board", &CandyCrushEnv::get_board)
        .def("get_score", &CandyCrushEnv::get_score)
        .def("get_steps", &CandyCrushEnv::get_steps)
        .def("get_max_steps", &CandyCrushEnv::get_max_steps)
        .def("get_target_score", &CandyCrushEnv::get_target_score)
        .def("get_last_swap", &CandyCrushEnv::get_last_swap)
        .def("get_last_reward_breakdown", &CandyCrushEnv::get_last_reward_breakdown)
        .def("get_board_string", &CandyCrushEnv::get_board_string);
}
