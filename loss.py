import torch
import numpy as np
import torch.autograd as autograd

def compute_td_loss(policy_net, target_net, batch_size, gamma, loss_fn,\
                                                         replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    USE_CUDA = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    Variable = lambda *args, **kwargs: autograd.Variable(*args, \
        **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
    state = Variable(Tensor(np.float32(state)), requires_grad=True)
    next_state = Variable(Tensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(np.float32(action)))
    reward = Variable(Tensor(np.float32(reward)))
    done = Variable(Tensor(np.float32(done)))

    # Implement the Temporal Difference Loss
    q_val = torch.gather(policy_net.forward(state), 1, action.unsqueeze(1))
    q_val_target = target_net.forward(next_state).max(1)[0]
    y_i = reward + gamma * (1 - done) * q_val_target 
    loss = loss_fn(y_i, q_val.squeeze())
    return loss