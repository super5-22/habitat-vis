import os
import git

import habitat_sim
from collections.abc import MutableMapping
from typing import Any, Dict, List
from typing import Optional, Union, cast, overload

from topdown import TopDownMap


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
# %cd $dir_path
data_path = os.path.join(dir_path, "data")


class HabitatVisSim(habitat_sim.Simulator):
    SENSOR2TYPE = {'color_sensor': 'color', 'depth_sensor': 'depth', 'semantic_sensor': 'semantic'}

    def __init__(self, cfg_dict, *args, **kwargs):
        super(HabitatVisSim, self).__init__(*args, **kwargs)
        self.obj_templates_mgr = self.get_object_template_manager()
        self.rigid_obj_mgr = self.get_rigid_object_manager()
        locobot_template_id = self.obj_templates_mgr.load_configs(
            str(os.path.join(data_path, "objects/locobot_merged"))
        )[0]

        self.locobot = []
        for i in range(1, len(self.agents)):
            # add robot object to the scene with the agent/camera SceneNode attached
            self.locobot.append(self.rigid_obj_mgr.add_object_by_template_id(
                locobot_template_id, self.agents[i].scene_node
            ))

        scene_bb = self.get_active_scene_graph().get_root_node().cumulative_bb
        x = scene_bb.x()
        z = scene_bb.z()
        print(f"The size of the scene is {x.max - x.min}m * {z.max - z.min}m")
        # navmesh_success = self.recompute_navmesh(
        #     self.pathfinder, navmesh_settings, include_static_objects=False
        # )
        # navmesh_save_path = "./412.navmesh"
        # self.pathfinder.save_nav_mesh(navmesh_save_path)
        self.control_agent_id = 0
        self.display_agent_id = 1
        self.display_name = "1"
        self.display_3rd_color = True
        self.AGENTID2NAME = {i: f"{i}" for i in range(1, len(self.agents))}
        self.AGENTID2NAME[0] = '3rd'
        self.previous_display_agent_id = 0
        assert len(self.agents) <= 10, "We do not support more than 9 agents since no keyboard mapping"
        # draw topdown_map costs a lot of time
        self._display_topdown = cfg_dict["topdown_map"]
        self._topdown = None
        if self._display_topdown:
            self._topdown = TopDownMap(self)
        self.user_mode = False

    def get_agent_state(self, agent_id=0):
        return self.get_agent(agent_id).get_state()

    def step(self, action, *args, **kwargs):
        if not isinstance(action, MutableMapping):
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
        observations = super().step(action, *args, **kwargs)
        observed_id = observations.keys()
        if self.display_agent_id not in observed_id:
            observations[self.display_agent_id] = self.get_sensor_observations(self.display_agent_id)
        if 0 not in observed_id:
            observations[0] = self.get_sensor_observations(0)
        if self._display_topdown:
            observations[self.display_agent_id]["top_down_map"] = self._topdown.update_metric(self.display_agent_id)

        return observations

    def reset(self, *args, **kwargs):
        agents_ids = [i for i in range(len(self.agents))]
        observations = super().reset(agents_ids)
        if self._display_topdown:
            self._topdown.reset_metric(None)
        return observations

    def o_func(self, key_button):
        self.control_agent_id = 0
        print("Change to 3rd view")

    def u_func(self, key_button):
        self.user_mode = True
        print("Turn on user_mode")

    def p_func(self, key_button):
        self.user_mode = False
        print("Turn off user_mode")

    def c_func(self, key_button):
        self.display_3rd_color = not self.display_3rd_color
        print("Change 3rd view display")

    def number_func(self, key_button):
        agent_id = int(chr(key_button))
        self.control_agent_id = agent_id
        self.display_agent_id = agent_id
        self.display_name = self.AGENTID2NAME[agent_id]
        print(f"Change to Agent {agent_id}")

    def change_display(self, keyboard_botton):
        self.previous_display_agent_id = self.display_agent_id
        if ord('0') < keyboard_botton < ord(f'{len(self.agents)}'):
            self.number_func(keyboard_botton)
        else:
            try:
                getattr(self, chr(keyboard_botton) + '_func')(keyboard_botton)
            except (ValueError, AttributeError):
                pass
        return self.display_agent_id == self.previous_display_agent_id
