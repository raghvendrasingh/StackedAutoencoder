function [ cost, grad ] = stackedAECost(theta,inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda,data,labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
%softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));

for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
a=cell(size(stack,1)+1,size(stack,2));
a{1}.data = data;
%size(a{1}.data)
for i=1:numel(stack)
   a{i+1}.data = sigmoid(stack{i}.w * a{i}.data + repmat(stack{i}.b,1,size(a{i}.data,2)));
end

N = softmaxTheta*a{i+1}.data;
N = bsxfun(@minus, N, max(N, [], 1));
N = exp(N);
N = bsxfun(@rdivide,N,sum(N));

temp = groundTruth .* log(N);


cost = (-1/size(data,2)).* sum(sum(temp)) + (lambda/2) .* sum(sum(softmaxTheta.^2)) ;
       % +(lambda/2) .* sum(sum(stack{2}.w.^2)) + (lambda/2) .* sum(sum(stack{1}.w.^2));
        


alp = groundTruth - N;

%size(N)
%size(alp)
%size(N)
%size(softmaxTheta)

%delta4 = ((-1 .* softmaxTheta' ) * alp) .* (N .* (1-N));
delta3 = ((-1 .* softmaxTheta' ) * alp) .* (a{3}.data .* (1-a{3}.data));
delta2 = (stack{2}.w' * delta3) .* (a{2}.data .* (1-a{2}.data));

%del_softmaxTheta = delta4 * a{3}.data';
del_w2 = delta3 * a{2}.data';
del_w1 = delta2 * a{1}.data';
del_b2 = sum(delta3,2);
del_b1 = sum(delta2,2);

softmaxThetaGrad = (-1/size(data,2)) .* (alp * a{3}.data') + lambda * softmaxTheta;


stackgrad{1}.w = (1/size(data,2)).* del_w1 ;%+ lambda .* stack{1}.w;
stackgrad{2}.w = (1/size(data,2)).* del_w2 ;%+ lambda .* stack{2}.w;
stackgrad{1}.b = (1/size(data,2)).* del_b1;
stackgrad{2}.b = (1/size(data,2)).* del_b2;













% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
