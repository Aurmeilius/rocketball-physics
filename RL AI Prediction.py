import json
import math
import sys
import os
import time
import subprocess
import urllib.request
import contextlib

#num_frames = 0

process_replay = False
train_on_data = False
grab_online = False
grab_online_count = 0

files = []
rl_data = []

for root, dirs, these_files in os.walk("."):
	for file_name in these_files:
		if file_name.endswith(".replay"):
			files.append(os.path.join(root, file_name))
		if file_name.endswith(".rldata"):
			rl_data.append(os.path.join(root, file_name))

if len(sys.argv) == 1:
	sys.exit("Use arguments -online, -replay and/or -train to specify program operation.")
elif len(sys.argv) > 1:
	for arg in sys.argv[1:]:
		if arg == "-online":
			grab_online = True
		if arg == "-replay":
			process_replay = True
		if arg == "-train":
			train_on_data = True

end_data_set = []

end_data_set.append([])
end_data_set.append([])

if grab_online:

	reached_end = False
	page_string = "https://www.rocketleaguereplays.com/api/replays"
	details = {"User-Agent": "Mozilla/5.0"}
	ex_replays = 0
	proper_indices = [1, 2, 3, 4, 10, 11, 12, 13]
	
	# Obtain data from RocketLeagueReplays API
	# TODO: accept replays post v1.35 as latest patch where ball physics changed, not just season 6

	while not reached_end:
		form_request = urllib.request.Request(page_string, None, headers=details)
		content = {}
		expect_half = 0
		
		with contextlib.closing(urllib.request.urlopen(form_request)) as url_handle:
			content = json.loads(url_handle.read())
	
		for replay in content["results"]:
			if replay["season"]["title"] != "Competitive Season 6": # Limit to season 6 unranked/ranked replays only, for now
				expect_half = expect_half + 1
				if expect_half == 30: # Raise flag when we have effectively seen no more season 6 replays
					reached_end = True
					break
				continue
			elif replay["playlist"] in proper_indices:
				files.append(replay["file"])

			#if replay["date_created"] > "2017-07-05T00:00:00Z": - v1.35 release date
				#print(replay["date_created"])
				#ex_replays = ex_replays + 1
			"""
			current_file = replay["file"]
			form_request_file = urllib.request.Request(current_file, None, headers=details)
			try:
				underproc = subprocess.run(["rattletrap", "-c", "-i", current_file], stdout=subprocess.PIPE, check=True, encoding="utf-8")
			except subprocess.CalledProcessError: # Rattletrap failed to process the replay, so copy it, and send to them later for the creator to debug.
				with open(str(time.time()) + ".replay", "wb") as writer:
					writer.write(urllib.request.urlopen(form_request_file).read())
					writer.close()
			"""
		page_string = content["next"]

	print("Replays from Online:", len(files))

if process_replay:
	if len(files) == 0:
		sys.exit("No replay files exist for processing.")

	for file_path in files:
		process = ""
		try:
			process = subprocess.run(["rattletrap", "-c", "-i", file_path], stdout=subprocess.PIPE, check=True, encoding="utf-8")
		except subprocess.CalledProcessError:
			print("Replay Failed:", file_path)
			continue

		main_data = json.loads(process.stdout)
		
		#num_frames = main_data["header"]["properties"]["value"]["NumFrames"]["value"]["int_property"]

		#print(main_data["content"]["frames"][1]["delta"])
		#print(main_data["header"]["properties"]["value"]["NumFrames"]["value"]["int_property"])
		"""
		with open("C:\\Users\\Aaron Mark Mugabe\\Desktop\TESTTWO.json", "w") as handme:
			handme.write(json.dumps(main_data, sort_keys=True, indent=4))
			handme.close()
		"""

		ball_value = -1 # The ball has not spawned
		ball_data = []
		ball_in_play = False
		goal_scored = False

		car_values = [] # No cars spawned
		car_position = []

		#ball_ticks = 0
		#hits = []

		old_no_contact = False

		for frame in main_data["content"]["body"]["frames"]:

			replicate_frame = frame["replications"]
			time_frame = frame["time"]

			for rep_item in replicate_frame:
				if "spawned" in rep_item["value"]: # Object Spawned In-Game
					if rep_item["value"]["spawned"]["class_name"] == "TAGame.Ball_TA" and (not ball_in_play) and (not goal_scored):
						ball_value = rep_item["actor_id"]["value"]
						ball_in_play = True
						#print("BALL RESPAWN", ball_value, time_frame)
					elif rep_item["value"]["spawned"]["class_name"] == "TAGame.Car_TA" and (rep_item["actor_id"]["value"] not in car_values):
						car_values.append(rep_item["actor_id"]["value"])
						car_position.append([])
						#print("CAR RESPAWN", car_values, car_position, time_frame)

				elif "destroyed" in rep_item["value"]: # Object Destroyed In-Game
					if rep_item["actor_id"]["value"] in car_values:
						delete_index = car_values.index(rep_item["actor_id"]["value"])
						del car_position[delete_index]
						del car_values[delete_index]
						#print("CAR DESTROYED", rep_item["actor_id"]["value"], time_frame)
					elif rep_item["actor_id"]["value"] == ball_value and goal_scored:
						ball_value = -1
						ball_in_play = False
						goal_scored = False
						#print("BALL DESTROYED", rep_item["actor_id"]["value"], time_frame)

				elif "updated" in rep_item["value"]: # Update Object Position/Rotation, but check if goal was scored first
					if rep_item["actor_id"]["value"] == ball_value:
						if rep_item["value"]["updated"][0]["name"] == "Engine.Actor:bCollideActors" and ball_in_play and (not goal_scored):
							ball_in_play = False
							goal_scored = True
							#print("GOAL", time_frame)

					if rep_item["actor_id"]["value"] == ball_value or (rep_item["actor_id"]["value"] in car_values):
						obj_struct = None

						# Now go for positional and rotational data

						for arr in rep_item["value"]["updated"]:
							if arr["name"] == "TAGame.RBActor_TA:ReplicatedRBState":
								obj_struct = arr["value"]["rigid_body_state"]
								break

						if obj_struct != None: # If fails, something else was updated, other than positional/rotational data (i.e, throttle or steer)
							if rep_item["actor_id"]["value"] != ball_value: # Fetch Car Data
								car_position[car_values.index(rep_item["actor_id"]["value"])] = [obj_struct["location"]["x"], obj_struct["location"]["y"], 
									obj_struct["location"]["z"], time_frame]
							elif rep_item["actor_id"]["value"] == ball_value and ball_in_play and (not goal_scored): # Fetch ball Data
								if ("linear_velocity" not in obj_struct) and ("angular_velocity" not in obj_struct): # Occurs during ball respawn
									ball_data = [obj_struct["location"]["x"], obj_struct["location"]["y"], obj_struct["location"]["z"],
										obj_struct["rotation"]["x"]["value"], obj_struct["rotation"]["y"]["value"], obj_struct["rotation"]["z"]["value"], 
										0, 0, 0, 0, 0, 0, time_frame]
								else:
									ball_data = [obj_struct["location"]["x"], obj_struct["location"]["y"], obj_struct["location"]["z"],
										obj_struct["rotation"]["x"]["value"], obj_struct["rotation"]["y"]["value"], obj_struct["rotation"]["z"]["value"], 
										obj_struct["linear_velocity"]["x"], obj_struct["linear_velocity"]["y"], obj_struct["linear_velocity"]["z"], 
										obj_struct["angular_velocity"]["x"], obj_struct["angular_velocity"]["y"], obj_struct["angular_velocity"]["z"], time_frame]
							
						"""
						if len(rep_item["value"]["updated"]) > 1:
							for i in range(len(rep_item["value"]["updated"])):
								if rep_item["value"]["updated"][i]["name"] == "TAGame.Ball_TA:HitTeamNum":
									hits.append(frame["time"])

						ball_ticks = ball_ticks + 1
						"""
			
			# A workaround that is close enough to a proper to a theoretical TAGame.Ball_TA:HitPlayerNum marking.
			# Using TAGame.Ball_TA:HitTeamNum is inadequate.

			if len(ball_data) > 0:
				car_hit_ball = False
				ball_data[12] = time_frame

				for car_pos in car_position:
					try:
						dist = math.sqrt(math.fsum([(car_pos[0] - ball_data[0]) ** 2, 
							(car_pos[1] - ball_data[1]) ** 2, 
							(car_pos[2] - ball_data[2]) ** 2]))
					except Exception as e: # An Index Out Of Range Exception BECAUSE CAR RESPAWNS IN REPLAYS ARE INCONSISTENT AF SO I'M SORRY
						#print(e,car_values,car_position,time_frame)
						break

					if dist < 250:
						car_hit_ball = True

				if not car_hit_ball and ball_in_play:
					if not old_no_contact:
						end_data_set[0].append(list(ball_data))
						end_data_set[1].append(list(ball_data))
						old_no_contact = True
					else:
						end_data_set[1][-1] = list(ball_data)
				else:
					old_no_contact = False

		#print("\nFinal Ball Pos\n------------------------\n", ball_data)
		#print("\nFinal Car Pos\n------------------------\n", car_position)
		print("Parsed", file_path, "- Current Data Points:", len(end_data_set[0]))

	for i in range(len(end_data_set[0])):
		end_data_set[0][i][-1] = end_data_set[1][i][-1] - end_data_set[0][i][-1]
		del end_data_set[1][i][-1]

	if not train_on_data:
		with open(str(time.time()) + ".rldata", "w") as dump:
			dump.write(json.dumps(end_data_set, separators=(',',':')))
			dump.close()

if train_on_data:
	if not process_replay:
		if len(rl_data) == 0:
			sys.exit("No data to train with exists.")

	if len(rl_data) > 0:
		for datum in rl_data:
			with open(datum, "r") as opened:
				temp = json.loads(opened.read())
				end_data_set[0].extend(temp[0])
				end_data_set[1].extend(temp[1])
				opened.close()

	print(len(end_data_set[0]))
	# TEMPORARY: Tweak final data set to see if results change:
	# What if only specifying distance is the key factor?

	for i in range(len(end_data_set[1])):
		end_data_set[1][i] = end_data_set[1][i][:3]

	# Start of Training Simulation with Tensorflow

	import tensorflow as tf
	import numpy

	# Based on ReLU
	# TODO: Take a look at PReLU, and see if it helps.

	def inital_weight_sqrt(shape):
		return tf.Variable(tf.random_normal(shape, stddev=numpy.sqrt(2.0/shape[0])))

	def initial_bias(shape):
		return tf.Variable(tf.constant(0.0, shape=shape))

	# Initial Approximation of train_bounds

	train_bounds = math.ceil(len(end_data_set[0]) * 0.9)
	least_common_multiple = 65536
	batch_size = 1024
	point_in_batch = 0

	# Training data divisible by chosen batch size, finalize train_bounds value
	
	while train_bounds % least_common_multiple != 0:
		train_bounds = train_bounds - 1

	data_input = numpy.asarray(end_data_set[0], numpy.float32)
	data_output = numpy.asarray(end_data_set[1], numpy.float32)

	numpy.random.seed(int(time.time()))
	rng_state = numpy.random.get_state()
	numpy.random.shuffle(data_input)
	numpy.random.set_state(rng_state)
	numpy.random.shuffle(data_output)
	
	# Multi-Layer NN: 13->...->(12 or 3, most likely 3))
	# Best NN so far: 13->300->300->300->300->300->300->300->300->3 with 193.16919 as error after 500 epochs. Batch Size = 1024
	# Figured out reason to previous unstability. Turns out initializing weights to a Normal dist. where the mean is nonzero and the std. dev. is constant regardless of inputs to neuron is a bad idea.

	x_input = tf.placeholder(tf.float32, shape=[None, 13])
	y_actual = tf.placeholder(tf.float32, shape=[None, 3])
	keep_prob = tf.placeholder(tf.float32)

	W_lay1 = inital_weight_sqrt([13, 300])
	b_lay1 = initial_bias([300])
	first_layer = tf.nn.relu(tf.add(tf.matmul(x_input, W_lay1), b_lay1))
	first_layer_drop = tf.nn.dropout(first_layer, keep_prob)

	W_lay2 = inital_weight_sqrt([300, 300])
	b_lay2 = initial_bias([300])
	second_layer = tf.nn.relu(tf.add(tf.matmul(first_layer_drop, W_lay2), b_lay2))
	second_layer_drop = tf.nn.dropout(second_layer, keep_prob)

	W_lay3 = inital_weight_sqrt([300, 300])
	b_lay3 = initial_bias([300])
	third_layer = tf.nn.relu(tf.add(tf.matmul(second_layer_drop, W_lay3), b_lay3))
	third_layer_drop = tf.nn.dropout(third_layer, keep_prob)

	W_lay4 = inital_weight_sqrt([300, 300])
	b_lay4 = initial_bias([300])
	fourth_layer = tf.nn.relu(tf.add(tf.matmul(third_layer_drop, W_lay4), b_lay4))
	fourth_layer_drop = tf.nn.dropout(fourth_layer, keep_prob)
	
	W_lay5 = inital_weight_sqrt([300, 300])
	b_lay5 = initial_bias([300])
	fifth_layer = tf.nn.relu(tf.add(tf.matmul(fourth_layer_drop, W_lay5), b_lay5))
	fifth_layer_drop = tf.nn.dropout(fifth_layer, keep_prob)
	
	W_lay6 = inital_weight_sqrt([300, 300])
	b_lay6 = initial_bias([300])
	sixth_layer = tf.nn.relu(tf.add(tf.matmul(fifth_layer_drop, W_lay6), b_lay6))
	sixth_layer_drop = tf.nn.dropout(sixth_layer, keep_prob)

	W_lay7 = inital_weight_sqrt([300, 300])
	b_lay7 = initial_bias([300])
	seventh_layer = tf.nn.relu(tf.add(tf.matmul(sixth_layer_drop, W_lay7), b_lay7))
	seventh_layer_drop = tf.nn.dropout(seventh_layer, keep_prob)

	W_lay8 = inital_weight_sqrt([300, 300])
	b_lay8 = initial_bias([300])
	eighth_layer = tf.nn.relu(tf.add(tf.matmul(seventh_layer_drop, W_lay8), b_lay8))
	eighth_layer_drop = tf.nn.dropout(eighth_layer, keep_prob)
	
	W_lay9 = inital_weight_sqrt([300, 300])
	b_lay9 = initial_bias([300])
	ninth_layer = tf.nn.relu(tf.add(tf.matmul(eighth_layer_drop, W_lay9), b_lay9))
	ninth_layer_drop = tf.nn.dropout(ninth_layer, keep_prob)

	W_lay10 = inital_weight_sqrt([300, 300])
	b_lay10 = initial_bias([300])
	tenth_layer = tf.nn.relu(tf.add(tf.matmul(ninth_layer_drop, W_lay10), b_lay10))
	tenth_layer_drop = tf.nn.dropout(tenth_layer, keep_prob)
	
	W_out = inital_weight_sqrt([300, 3])
	b_out = initial_bias([3])

	y_calc = tf.add(tf.matmul(tenth_layer_drop, W_out), b_out)

	distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_actual, y_calc)), 1)))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(distance)

	# Add saver, so models can be saved/loaded without rerun
	
	epoch_counter = 0

	save_state = tf.train.Saver()

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		"""
		try:
			save_state.restore(session, ".\BallPhysicsModel")
		except Exception: # No existing model was found, no problem.
			pass
		"""
		while epoch_counter < 500:
			train_input = data_input[point_in_batch:point_in_batch + batch_size]
			train_output = data_output[point_in_batch:point_in_batch + batch_size]
			point_in_batch = point_in_batch + batch_size

			if point_in_batch == train_bounds: # Completed an epoch (i.e all training data has seen the network with the same frequency.)
				point_in_batch = 0
				epoch_counter = epoch_counter + 1
				print("Epoch", epoch_counter, "- Error:", distance.eval(feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1}))

			session.run(train_step, feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1})
			
		#save_state.save(session, ".\BallPhysicsModel")
		print("Final Error:", distance.eval(feed_dict={x_input:data_input[train_bounds:], y_actual:data_output[train_bounds:], keep_prob: 1}))
		
		session.close()

"""
print("Ball Instances: %d Num Frames: %d" % (ball_ticks, num_frames))
print("\nHit Ball Timestamps (", len(hits), ")\n------------------------\n", hits)
print("\nFinal Data Set (", len(end_data_set[0]), ")\n------------------------\n", end_data_set)
"""
