# tree.grow
# 
# Args:
#  x: Training data matrix.
#  y: Vector containing binary class labels.
#  nmin: The minimum number of cases a node should contain for it to be allowed to be split.
#  minleaf: The minimum number of cases leaf nodes should contain.
#  nfeat: The number of (random) features that should be considered for each split.
# 
# Returns:
#  A list structure that represents a classification tree.
# 
# Grow a classification tree on a set of training data.
tree.grow <- function(x, y, nmin, minleaf, nfeat) {
  # Create a list to hold the tree with the initial root node.
  nodes <- list(list(left=NULL, right=NULL, impurity=impurity(y), is_leaf=FALSE, class=NULL, indices=1:nrow(x), feature=NULL, threshold=NULL))
  
  # Indices of all unhandled nodes.
  unhandled_nodes = c(1)
  
  while (length(unhandled_nodes) > 0) {
    # Get the first unhandled node.
    node_index <- unhandled_nodes[1]
    
    if (length(unhandled_nodes) > 1)
      unhandled_nodes <- unhandled_nodes[2:length(unhandled_nodes)]
    else
      unhandled_nodes <- c()
    
    if (nodes[[node_index]]$impurity > 0) {
      
      # If the node does not contain enough observations, don't split it.
      if (length(nodes[[node_index]]$indices) < nmin) {
        # Mark the node as leaf node.
        nodes[[node_index]]$is_leaf <- TRUE
        # Assign the majority class.
        nodes[[node_index]]$class <- determine_majority_class(y, nodes[[node_index]]$indices)
        # Set the indices to NULL, as they are not needed anymore.
        nodes[[node_index]]$indices <- NULL
        
        next
      }
      
      # Select random features to split on.
      features <- sample(1:ncol(x), nfeat, replace=FALSE)
      
      # Determine feature and threshold that results in the best split.
      best_split <- list("feature"=NULL, "impurity_reduction"=.Machine$double.xmin, "impurity_left"=0, "impurity_right"=0, "threshold"=0, "index"=0, "order"=NULL)
      for (f in features) {
        attribute_data <- x[,f][nodes[[node_index]]$indices]
        class_data <- y[nodes[[node_index]]$indices]
        
        split <- determine_best_split(attribute_data, class_data, nodes[[node_index]]$impurity, minleaf)
        
        if (split$impurity_reduction > best_split$impurity_reduction) {
          best_split$feature <- f
          best_split$impurity_reduction <- split$impurity_reduction
          best_split$impurity_left <- split$impurity_left
          best_split$impurity_right <- split$impurity_right
          best_split$threshold <- split$threshold
          best_split$index <- split$index
          best_split$order <- split$order
        }
      }
      
      if (!is.null(best_split$feature)) {
        # Set the left and right child indices of the current node.
        nodes[[node_index]]$left <- length(nodes) + 1
        nodes[[node_index]]$right <- length(nodes) + 2
        # Set the decision feature and threshold of the current node.
        nodes[[node_index]]$feature <- best_split$feature
        nodes[[node_index]]$threshold <- best_split$threshold
        
        indices <- nodes[[node_index]]$indices[best_split$order]
        indices_left <- indices[1:best_split$index]
        indices_right <- indices[(best_split$index + 1):length(indices)]
        
        # Create new left and right node.
        left_node <- list("left"=NULL, "right"=NULL, "impurity"=best_split$impurity_left, "is_leaf"=FALSE, "class"=NULL, "indices"=indices_left, "feature"=NULL, "threshold"=NULL)
        right_node <- list("left"=NULL, "right"=NULL, "impurity"=best_split$impurity_right, "is_leaf"=FALSE, "class"=NULL, "indices"=indices_right, "feature"=NULL, "threshold"=NULL)
        
        # Append new nodes to tree.
        nodes[[length(nodes) + 1]] <- left_node
        nodes[[length(nodes) + 1]] <- right_node
        
        # Mark new nodes as unhandled.
        unhandled_nodes <- c(unhandled_nodes, nodes[[node_index]]$left, nodes[[node_index]]$right)
      }
      else {
        # Mark the node as leaf node.
        nodes[[node_index]]$is_leaf <- TRUE
        # Assign the majority class.
        nodes[[node_index]]$class <- determine_majority_class(y, nodes[[node_index]]$indices)
        # Set the indices to NULL, as they are not needed anymore.
        nodes[[node_index]]$indices <- NULL
      }
    }
    else {
      # Mark the node as leaf node.
      nodes[[node_index]]$is_leaf <- TRUE
      # Assign the majority class.
      nodes[[node_index]]$class <- determine_majority_class(y, nodes[[node_index]]$indices)
      # Set the indices to NULL, as they are not needed anymore.
      nodes[[node_index]]$indices <- NULL
    }
  }
  
  return(nodes)
}

# tree.classify
# 
# Args:
#  x: Data matrix to be classified.
#  tr: Classification tree.
# 
# Returns:
#  A vector of predicted class labels.
# 
# Predict the classes for a collection of data.
tree.classify <- function(x, tr) {
  y <- c()
  for(case in 1:nrow(x)){
    node_index <- 1
    
    #while not in a leaf node determine which child node to traverse to
    while (!tr[[node_index]]$is_leaf){
      feature <- tr[[node_index]]$feature
      threshold <- tr[[node_index]]$threshold
      value <- x[case, feature]
      #determine the child node based on the threshold
      if(value <= threshold){
        node_index <- tr[[node_index]]$left
      }
      else{
        node_index <- tr[[node_index]]$right
      }
    }
    #append the classification of the leaf node to our return set
    y <- c(y, tr[[node_index]]$class)
  }
  
  return(y)
}

# tree.grow.bag
# 
# Args:
#  x: Training data matrix.
#  y: Vector containing binary class labels.
#  nmin: The minimum number of cases a node should contain for it to be allowed to be split.
#  minleaf: The minimum number of cases leaf nodes should contain.
#  nfeat: The number of (random) features that should be considered for each split.
#  m: The number of (randomized) trees that have to be created.
# 
# Returns:
#  A list of classification trees.
# 
# Grow multiple classification trees on a set of training data, using bagging.
tree.grow.bag <- function(x, y, nmin, minleaf, nfeat, m) {
  trees <- list()
  for (i in 1:m) {
    samples <- sample(1:nrow(x), nrow(x), replace=TRUE)
    # Grow a new tree with m random samples from x.
    trees[[i]] <- tree.grow(x[samples,], y[samples], nmin, minleaf, nfeat)
  }
  return(trees)
}

# tree.classify.bag
# 
# Args:
#  x: Data matrix to be classified.
#  trees: List of classification trees.
# 
# Returns:
#  A vector of predicted class labels.
# 
# Predict the classes for a collection of data using a list of (bagging) trees.
tree.classify.bag <- function(x, trees) {
  classes <- matrix(0, nrow=nrow(x), ncol=length(trees))
  
  # Predict the class with each tree.
  for (i in 1:length(trees)) {
    classes[,i] <- tree.classify(x, trees[[i]])
  }
  
  majority_classes <- c()
  # Determine the majority class for each row.
  for (i in 1:nrow(x)) {
    if (sum(classes[i,] == 1) > sum(classes[i,] == 0))
      majority_classes <- c(majority_classes, 1)
    else
      majority_classes <- c(majority_classes, 0)
  }
  
  return(majority_classes)
}

# impurity
#
# Args:
#  x: Vector containing binary class labels.
# 
# Returns:
#  Impurity of vector.
#
# Calculate the impurity (using Gini) of a vector of binary class labels.
impurity <- function(x) {
  p <- sum(x == 0) / length(x)
  return(p * (1-p))
}

# get_split_indices
#
# Args:
#  x: Vector containing numeric attributes.
#  minleaf: Integer defining the minimum number of observations in each leaf node.
# 
# Returns:
#  Vector of indices.
#
# For a sorted list of numeric attributes, get the indices of all possible splits.
get_split_indices <- function(x, minleaf) {
  indices <- rep(0, length(unique(x)) - 1)
  j <- 1
  for (i in 2:length(x)) {
    if (x[i] != x[i - 1]) {
      indices[j] = i - 1
      j <- j + 1
    }
  }
  
  # Return the splits with all indices that do not satisfy the minleaf constraint removed.
  return(indices[(indices >= minleaf) & (indices <= (length(x) - minleaf))])
}

# determine_majority_class
# 
# Args:
#  classes: Vector containing binary class labels.
#  indices: Vector containing indices for the classes vector. If NULL, the majority class in the entire classes vector is determined.
# 
# Returns:
#  The majority class (0 or 1).
# 
# Determine the majority class for a vector of binary class labels, possibly limited by a vector of indices. In the case of a tie class 0 is returned.
determine_majority_class <- function(classes, indices=NULL) {
  if (is.null(indices)) {
    if (sum(classes == 0) >= length(classes) / 2)
      return(0)
  }
  else {
    if (sum(classes[indices] == 0) >= length(indices) / 2)
      return(0)
  }
  
  return(1)
}

# determine_best_split
# 
# Args:
#  x: Vector containing attribute values.
#  y: Vector containing binary class labels.
#  impurity_parent: Impurity of the parent node.
#  minleaf: The minimum number of cases leaf nodes should contain.
# 
# Returns:
#  A list containing the impurity reduction, the impurity of the left and right node, the index and threshold of the split and the order of the attributes.
# 
# Determine the best allowed split on a node (if one exists).
determine_best_split <- function(x, y, impurity_parent, minleaf) {
  # Sort the elements pairwise by attribute value.
  o <- order(x)
  x <- x[o]
  y <- y[o]
  
  # Get all split indices that have to be evaluated.
  split_indices <- get_split_indices(x, minleaf)
  
  if (length(split_indices) == 0)
    return(list("impurity_reduction"=.Machine$double.xmin))
  
  impurity_reduction <- rep(0.0, length(split_indices))
  impurity_left <- rep(0.0, length(split_indices))
  impurity_right <- rep(0.0, length(split_indices))
  j <- 1
  
  # Calculate the impurity score for each split.
  for (i in split_indices) {
    # Fraction of observations that are sent to the left node.
    f <- i / length(x)
    # Calculate the impurity of the left node.
    il <- impurity(y[1:i])
    impurity_left[j] <- il
    # Calculate the impurity of the right node.
    ir <- impurity(y[(i+1):length(y)])
    impurity_right[j] <- ir
    # Calculate the impurity reduction.
    impurity_reduction[j] <- impurity_parent - (f * il + (1.0 - f) * ir)
    
    j <- j + 1
  }
  
  best_index <- which.max(impurity_reduction)
  index <- split_indices[best_index]
  threshold <- (x[index] + x[index + 1]) / 2.0
  
  return(list("index"=index, "threshold"=threshold, 
              "impurity_reduction"=max(impurity_reduction),
              "impurity_left"=impurity_left[best_index], "impurity_right"=impurity_right[best_index],
              "order"=o))
}