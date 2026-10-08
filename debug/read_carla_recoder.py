import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.reload_world()

summary=client.show_recorder_file_info("record_scenario_temp.log", True)
print(f"=============summary=========\n{summary}")
client.set_replayer_time_factor(1)
client.replay_file("record_scenario_temp.log", 1, 0, 16, True)

