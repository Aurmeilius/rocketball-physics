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
files_online = []
rl_data = []

for root, dirs, these_files in os.walk("."):
	for file_name in these_files:
		if file_name.endswith(".replay"):
			files.append(os.path.join(root, file_name))
		if file_name.endswith(".rldata"):
			rl_data.append(os.path.join(root, file_name))

if len(sys.argv) == 1:
	sys.exit("Use arguments -online, -fetch and/or -train to specify program operation.")
elif len(sys.argv) > 1:
	for arg in sys.argv[1:]:
		if arg == "-online":
			grab_online = True
		if arg == "-fetch":
			process_replay = True
		if arg == "-train":
			train_on_data = True

end_data_set = []

if grab_online:

	reached_end = False
	page_string = "https://www.rocketleaguereplays.com/api/replays"
	details = {"User-Agent": "Mozilla/5.0"}
	threshold_row = 0
	proper_playlists = [1, 2, 3, 4, 10, 11, 12, 13] # Unranked 1v1-4v4 and Ranked Modes 1v1,2v2,3v3solo,3v3
	failed_count = 0

	# Obtain data from RocketLeagueReplays API
	# Accept replays v1.35 and up as latest patch where ball physics changed, not just season 6
	# Turns out start of season 5 is patch v1.35

	while not reached_end:
		form_request = urllib.request.Request(page_string, None, headers=details)
		content = {}
		
		with contextlib.closing(urllib.request.urlopen(form_request)) as url_handle:
			content = json.loads(url_handle.read())
	
		for replay in content["results"]:
			if replay["season"]["title"] != "Competitive Season 6": # Limit to season X unranked/ranked replays only, for now
				if replay["season"]["title"] < "Competitive Season 6" or replay["season"]["title"] == "Season 1": # Season X and below is not accepted. SX+1 and above is fine
					threshold_row = threshold_row + 1
					if threshold_row == 300: # Raise flag when we have effectively seen no more season X replays
						reached_end = True
						break
				else:
					threshold_row = 0
			elif replay["playlist"] in proper_playlists:
				files_online.append(replay["file"])
				threshold_row = 0
			else:
				failed_count = failed_count + 1

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

	print("Replays from Online:", len(files_online), "-", failed_count)

if process_replay:
	def process_replay_and_append_data(main_data):
		ball_value = -1
		ball_data = []
		ball_in_play = False
		goal_scored = False

		car_values = []
		car_position = []

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
						continue

					if dist < 250: # Farthest dist while still hitting the ball is 173.124233uu (tested w/ breakout), add buffer to that due to "replay lag" to be on the safe side.
						car_hit_ball = True

				if not car_hit_ball and ball_in_play:
					if not old_no_contact:
						end_data_set.append([])
						end_data_set[-1].append(ball_data)
						old_no_contact = True
					else:
						end_data_set[-1].append(ball_data)
				else:
					old_no_contact = False

	if len(files) == 0 and len(files_online) == 0:
		sys.exit("No replay files exist for processing.")

	files.extend(files_online)

	for file_path in files:
		process = ""
		try:
			process = subprocess.run(["rattletrap", "-c", "-i", file_path], stdout=subprocess.PIPE, check=True, encoding="utf-8")
		except subprocess.CalledProcessError:
			print("Replay Failed:", file_path)
			#files.append(file_path)
			continue

		main_data = json.loads(process.stdout)
		regular_match = False
		packages = main_data["content"]["body"]["packages"]
		map_name = main_data["header"]["body"]["properties"]["value"]["MapName"]["value"]["name"]
		mutator_flags = ["..\\..\\TAGame\\CookedPCConsole\\Mutators_SF.upk", 
						"..\\..\\TAGame\\CookedPCConsole\\Mutators_Balls_SF.upk", 
						"..\\..\\TAGame\\CookedPCConsole\\Mutators_Items_SF.upk"]
		non_standard_maps = ["NeoTokyo_P", "ARC_P", "Wasteland_P", "Wasteland_Night_P"]

		if "..\\..\\TAGame\\CookedPCConsole\\GameInfo_Soccar_SF.upk" in packages: # Playing in Soccar Mode
			if len([pack for pack in packages if pack in mutator_flags]) == 0: # No Mutators
				if map_name not in non_standard_maps and "Labs" not in map_name: # Not A Non-Standard or Rocket Labs Map
					regular_match = True
			
		if regular_match:
			process_replay_and_append_data(main_data)
			print("Parsed", file_path, "- Current Data Points:", len(end_data_set))
		else:
			print(file_path, "does not meet the standard requirements.")

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

	# General Multi-Layer Perception Graph

	def create_graph(given_input, layers): # TODO: Add Dropout
		tensor_weights = []
		tensor_bias = []
		tensor_layers = []

		for i in range(len(layers) - 1):
			tensor_weights.append(inital_weight_sqrt([layers[i], layers[i + 1]]))
			tensor_bias.append(initial_bias([layers[i + 1]]))

			if i + 2 == len(layers):
				if i == 0:
					tensor_layers.append(tf.add(tf.matmul(given_input, tensor_weights[i]), tensor_bias[i]))
				else:	
					tensor_layers.append(tf.add(tf.matmul(tensor_layers[i - 1], tensor_weights[i]), tensor_bias[i]))
			elif i != 0: 
				tensor_layers.append(tf.nn.relu(tf.add(tf.matmul(tensor_layers[i - 1], tensor_weights[i]), tensor_bias[i])))	
			else:
				tensor_layers.append(tf.nn.relu(tf.add(tf.matmul(given_input, tensor_weights[i]), tensor_bias[i])))	

		return tensor_layers[-1]	

	# Initial Approximation of train_bounds

	train_bounds = math.ceil(len(end_data_set[0]) * 0.95)
	least_common_multiple = 2048
	batch_size = 2048
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
	# Best NN so far: 13->300->300->300->300->300->300->3 with 161.94354 as error after 200 epochs. Batch Size = 512
	# (13->500*10->3) with 148.25333 as error after 200 epochs, batch size = 2048, 
	# Figured out reason to previous unstability. Turns out initializing weights to a Normal dist. where the mean is far from zero and the std. dev. is constant regardless of inputs to neuron is a bad idea.

	x_input = tf.placeholder(tf.float32, shape=[None, 13])
	y_actual = tf.placeholder(tf.float32, shape=[None, 3])
	keep_prob = tf.placeholder(tf.float32)

	"""
	W_lay1 = inital_weight_sqrt([13, 400])
	b_lay1 = initial_bias([400])
	first_layer = tf.nn.relu(tf.add(tf.matmul(x_input, W_lay1), b_lay1))
	first_layer_drop = tf.nn.dropout(first_layer, keep_prob)
	"""
	
	y_calc = create_graph(x_input, [13, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 3])

	counter_out = 0
	epoch_counter = tf.Variable(counter_out, trainable=False)
	epoch_limit = 400

	rate_decay = tf.train.exponential_decay(0.001, epoch_counter, 110, 0.1, staircase=True)
	distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_actual, y_calc)), 1)))

	# Because Staired Exponential Decay Is Stubborn to Work
	train_step = tf.train.AdamOptimizer(1e-3).minimize(distance)
	train_step_small = tf.train.AdamOptimizer(1e-4).minimize(distance)
	train_step_smaller = tf.train.AdamOptimizer(1e-5).minimize(distance)
	train_step_smallest = tf.train.AdamOptimizer(1e-6).minimize(distance)

	# Add saver, so model(s) can be saved/loaded without rerun

	save_state = tf.train.Saver([y_calc])

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		"""
		try:
			save_state.restore(session, ".\BallPhysicsModel")
		except Exception: # No existing model was found, no problem.
			pass
		"""
		while counter_out < epoch_limit:
			train_input = data_input[point_in_batch:point_in_batch + batch_size]
			train_output = data_output[point_in_batch:point_in_batch + batch_size]

			if counter_out < 200:
				session.run(train_step, feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1})
			elif counter_out < 300:
				session.run(train_step_small, feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1})
			elif counter_out < 350:
				session.run(train_step_smaller, feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1})
			else:
				session.run(train_step_smallest, feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1})

			point_in_batch = point_in_batch + batch_size

			if point_in_batch == train_bounds: # Completed an epoch (i.e all training data has seen the network with the same frequency.)
				point_in_batch = 0
				counter_out = counter_out + 1
				epoch_counter = epoch_counter + 1
				print("Epoch", session.run(epoch_counter), "- Error:", distance.eval(feed_dict={x_input: train_input, y_actual: train_output, keep_prob: 1}))

		#save_state.save(session, ".\BallPhysicsModel")
		print("Final Error:", distance.eval(feed_dict={x_input:data_input[train_bounds:], y_actual:data_output[train_bounds:], keep_prob: 1}))
		
		session.close()
