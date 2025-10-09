# PEM
- conda activate pem_ss
- 训练 python train_net_neusoft(or mark).py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/pem_R50_bs32_90k.yaml MODEL.WEIGHTS output/model

# V1 区分车道线与道路标记 & 增加路沿与护栏
    classes          IoU     
    ---------------------
    car           : 0.945 车辆      
    human         : 0.834 行人    
    road          : 0.925 道路     
    lane_mark     : 0.439 车道线     
    curb          : 0.727 路沿     
    road_mark     : 0.575 道路标记     
    guard_rail    : 0.723 护栏     
    ---------------------
    Score Average : 0.738

# V2 护栏补充隔离带（Separator）类别 + 车道线实例化
    car           : 0.942      
    human         : 0.822      
    road          : 0.921      
    lane_mark     : 0.420      
    curb          : 0.709      
    road_mark     : 0.548      
    guard_rail    : 0.712      
    --------------------------------
    Score Average : 0.725

# V3 增加交通标志牌 + 黄白实虚线区分         
    car           : 0.943        
    human         : 0.832                                                                                             
    road          : 0.908
    lane_mark     : 0.395
    curb          : 0.716
    road_mark     : 0.551
    guard_rail    : 0.757
    traffic_sign  : 0.805                                                                                  
--------------------------------                                                                           
    Score Average : 0.738      

# V3_1
    car           : 0.941      
    human         : 0.829      
    road          : 0.917      
    lane_mark     : 0.407      
    curb          : 0.697      
    road_mark     : 0.557      
    guard_rail    : 0.755      
    traffic_sign  : 0.808      
    ----------------------------
    Score Average : 0.739      
    ----------------------------


# V4 道路标线细分语义
    box_junction        : 0.903      
    crosswalk           : 0.858      
    stop_line           : 0.697      
    solid_single_white  : 0.475      
    solid_single_yellow : 0.760      
    solid_single_red    : 0.613      
    solid_double_white  : 0.820      
    solid_double_yellow : 0.868      
    dashed_single_white : 0.701      
    dashed_single_yellow: 0.699      
    left_arrow          : 0.517      
    straight_arrow      : 0.594      
    right_arrow         : 0.193      
    left_straight_arrow : 0.542      
    right_straight_arrow: 0.552      
    channelizing_line   : 0.846      
    motor_prohibited    : 0.855      
    slow                : 0.834      
    motor_priority_lane : 0.675      
    motor_waiting_zone  : 0.739      
    left_turn_box       : 0.539      
    motor_icon          : 0.542      
    bike_icon           : 0.556      
    parking_lot         : 0.650      
    --------------------------------
    Score Average       : 0.668      

# V5 道路标线语义合并
    crosswalk           : 0.866      
    stop_line           : 0.745      
    solid_single_white  : 0.767      
    solid_single_yellow : 0.358      
    solid_double_white  : 0.827      
    solid_double_yellow : 0.872      
    dashed_single_white : 0.712      
    dashed_single_yellow: 0.744      
    arrow               : 0.819      
    --------------------------------
    Score Average : 0.746

# V6 Apollo道路标线
    classes          IoU     
    --------------------------------
    background    : 0.996      
    s_w_d         : 0.763      
    s_y_d         : 0.821      
    ds_y_dn       : 0.571      
    sb_w_do       : 0.324      
    sb_y_do       : 0.299      
    b_w_g         : 0.677      
    s_w_s         : 0.645      
    s_w_c         : 0.790      
    s_y_c         : 0.788      
    s_w_p         : 0.595      
    c_wy_z        : 0.767      
    a_w_u         : 0.736      
    a_w_t         : 0.790      
    a_w_tl        : 0.740      
    a_w_tr        : 0.766      
    a_w_l         : 0.783      
    a_w_r         : 0.700      
    a_n_lu        : 0.000      
    b_n_sr        : 0.393      
    d_wy_za       : 0.000      
    r_wy_np       : 0.000      
    vom_wy_n      : 0.618      
    om_n_n        : 0.361      
    --------------------------------
    Score Average : 0.580      
    --------------------------------