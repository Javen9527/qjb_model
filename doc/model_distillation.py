##################################
# Teacher：大模型、训好的、慢、强
# Student：小模型、随机初始化、要训练
# Dataset：你的训练数据（图 / 轨迹都行）
##################################

# 1. 加载训好的教师模型（输入1）
teacher = DiffusionUNet()
teacher.load_pretrained("teacher_ckpt.pth")
teacher.eval()       # 推理模式
teacher.requires_grad_(False)  # 冻结，绝对不训练

# 2. 初始化学生模型（将要被蒸馏出来）
student = SmallDiffusionUNet()  # 更小、更快

# 3. 训练数据（输入2）
dataloader = get_dataloader()

# 4. 训练核心
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
for x_0 in dataloader:
    # ==========================
    # 步骤1：构造加噪样本 x_t
    # ==========================
    t = sample_time_step()      # 随机采样时间步
    noise = torch.randn_like(x_0)
    x_t = add_noise(x_0, noise, t)  # 加噪

    # ==========================
    # 步骤2：教师模型给出“标准答案”
    # ==========================
    with torch.no_grad():
        noise_pred_teacher = teacher(x_t, t)

    # ==========================
    # 步骤3：学生模型去模仿教师
    # ==========================
    noise_pred_student = student(x_t, t)

    # ==========================
    # 步骤4：蒸馏损失（模仿老师）
    # ==========================
    loss = F.mse_loss(noise_pred_student, noise_pred_teacher)

    # ==========================
    # 步骤5：只更新学生
    # ==========================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. 输出结果
torch.save(student.state_dict(), "student_model.pth")


## 蒸馏优势：达到相同（掉点较低）的效果，但是速度／参数量都快很多（注意模型参数是完全不一样的了）