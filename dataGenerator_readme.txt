class DataGenerator():
	variables:
		self.dress_type      :  当前所要训练的服装类别
		self.joints_list     :  每个关节点的名称形成的list
		self.img_dir         :  训练图像所在文件夹（不包括/Image/***）
		self.train_data_file :  训练图像标签文件（包括文件夹目录）
		self.images          :  图像文件夹中所有文件名，#目前没用到#
		self.train_table     :  训练图像名称list
		self.data_dict       :  图像关键点标签字典，存储格式：
					data_dict[name] = {'box':box, 'joints':joints, 'weight':weight}
					其中，box = [box_left_up_x, box_left_up_y, box_right_down_x, box_right_down_y]
					        为图像中人体的裁剪框，根据最外围的关键点确定
					      joints = [[***_x, ***_y], [***_x, ***_y], ...]
					        存储关键点坐标，顺序与joints_list一致
					      weight = [len = len(joints),value = 1 or 0]
						表示对应关键点是否可见（存在），与标签的1对应
		self.letter          :  没研究明白是做什么用的。。。。在open_img里面用的
		self.train_set       :  选出的训练集
		self.valid_set       :  选出的验证集
	
	functions:
		__init__(dress_type, joints_list, img_dir, train_data_file)
			:  初始化
		test(toWait=0.2)              
			:  用来测试函数，可以在原图上画出对应map
		_read_train_data()   
			:  读取csv数据，对应存到train_table和data_dict中
		_create_set(validation_rate=0.1)        
			:  选出train_set和valid_set
		_give_batch_name(batch_size=16, set='train')
			:  选出一个batch的数据
		_generate_hm(height, width, joints, maxlength, weight)
			:  生成对应关节点的map，height、width为map的尺寸；
			   mxlength为bounding box的长度；
			   weight为关节点对应权重。
			调用_make_gaussian()生成高斯核
		_crop_data(height, width, box, boxp=0.05)
			:  一种裁剪（包括padding参数）图像的方法
		_crop_img(img, padding, crop_box)
			:  调用_crop_data()裁剪图片
		_crop(img, padding, crop_box)
			:  同时裁剪图片和对应map
		_augment(img, hm, max_rotation=30)
			:  图像旋转增强方法
		_aux_generator(batch_size=16, stacks=4, normalize=True, sample_set='train')
			:  调用之前的函数生成数据，不过实际程序没有用它
		  
		
		
