import csv
import math

class Mushroom:
    """
           Initializes a Mushroom object.

           Args:
               edible (bool): Indicates whether the mushroom is edible (True) or not (False).

           Attributes:
               edible (bool): Indicates whether the mushroom is edible.
               attributes (dict): Dictionary to store mushroom attributes.
           """
    def __init__(self, edible: bool):
        self.edible = edible
        self.attributes = {}
    def is_edible(self):
        return self.edible
    def add_attribute ( self , name : str , value : str ):
        self.attributes[name] = value
    def get_attribute(self, name: str):
        return self.attributes[name]
class Node:
    def __init__(self, criterion: str, is_leaf: bool = False):
        """
                Initializes a Node object for the decision tree.

                Args:
                    criterion (str): The criterion used for splitting at this node.
                    is_leaf (bool, optional): Indicates whether the node is a leaf node (True) or not (False).
                                              Defaults to False.

                Attributes:
                    criterion (str): The criterion used for splitting at this node.
                    is_leaf (bool): Indicates whether the node is a leaf node.
                    edges_ (list): List of edges connected to this node.
        """
        self.criterion_ = criterion
        self.leaf = is_leaf
        self.edges_ = []
    def is_leaf(self):
        return self.leaf
    def add_edge(self, label: str, child):
        self.edges_.append(Edge(self, child, label))
class Edge:
    def __init__(self, parent: Node, child: Node, label: str):
        """
                Initializes an Edge object for the decision tree.

                Args:
                    parent (Node): The parent node from which the edge originates.
                    child (Node): The child node to which the edge leads.
                    label (str): Label indicating the value of the splitting criterion for this edge.

                Attributes:
                    parent (Node): The parent node from which the edge originates.
                    child (Node): The child node to which the edge leads.
                    label (str): Label indicating the value of the splitting criterion for this edge.
                """
        self.parent_ = parent
        self.child_ = child
        self.label_ = label


def load_dataset(path: str):
    """
       Loads mushroom data from the specified CSV file and puts the mushroom object in a list.

       Args:
           path (str): File path to the CSV file containing mushroom data.

       Returns:
           list[Mushroom]: List of Mushroom objects representing the loaded data.

       """
    data = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader) # only takes the first line of mushrooms.csv

        for row in csvreader:
            is_edible = row[0] == 'Yes' # is edible == True if is row[0] == 'Yes else is edible == False
            m = Mushroom(is_edible)
            for i, attribute_value in enumerate(row):
                m.add_attribute(header[i], attribute_value)
            data.append(m)
    return data



def entropy(p_Y: float):
    """
        Calculates the entropy with probability p_Y

        Args:
            p_Y (float): Probability of success for the mushroom to be edible

        Returns:
            float: Entropy value.

        """
    if p_Y == 0 or p_Y == 1:
        return 0
    else:
        return -p_Y * math.log2(p_Y) - (1 - p_Y) * math.log2(1 - p_Y)

def information_gain(parent_entropy: float, subsets: list):
    """
       Calculates the information gain with the formula.

       Args:
           parent_entropy (float): Entropy of the parent node.
           subsets (list): List of subsets containing counts and probabilities.

       Returns:
           float: Information gain value.

       """
    total_weighted_entropy = sum(subset['count'] * entropy(subset['p_Y']) for subset in subsets)
    total_instances = sum(subset['count'] for subset in subsets)
    return parent_entropy - total_weighted_entropy / total_instances

def calculates_p_y(mushrooms: list):
    """
      Calculates the probability of having an edible mushroom (P_Y)

      Args:
          mushrooms (list[Mushroom]): List of Mushroom objects representing the dataset.

      Returns:
          float: Entropy value of the majority class.

      """
    count_edible = sum(1 for mushroom in mushrooms if mushroom.is_edible())
    count_total = len(mushrooms)
    p_Y = count_edible / count_total
    return entropy(p_Y)

def calculate_information_gain_for_attribute(mushrooms: list, attribute: str):
    """
      Calculates the information gain for a specific attribute.

      Args:
          mushrooms (list[Mushroom]): List of Mushroom objects representing the dataset.
          attribute (str): Attribute for which the information gain is calculated.

      Returns:
          float: Information gain value.

      """
    parent_entropy = calculates_p_y(mushrooms)
    subsets = []

    unique_values = set(m.get_attribute(attribute) for m in mushrooms)
    for value in unique_values:
        subset = [m for m in mushrooms if m.get_attribute(attribute) == value]
        count = len(subset)
        count_edible = sum(1 for m in subset if m.is_edible())
        p_Y = count_edible / count if count > 0 else 0
        subsets.append({'count': count, 'p_Y': p_Y}) #subsets contains the P_Y of each label

    return information_gain(parent_entropy, subsets)

def choose_best_attribute(mushrooms):
    """
        Chooses the best attribute for splitting the dataset based on information gain.

        Args:
            mushrooms (list[Mushroom]): List of Mushroom objects representing the dataset.

        Returns:
            tuple: Best attribute and its information gain value.

        """
    attributes = list(mushrooms[0].attributes.keys())[1:]
    best_attribute, best_gain = max(((attribute, calculate_information_gain_for_attribute(mushrooms, attribute)) for attribute in attributes), key=lambda x: x[1]) #The parameter key=lambda x: x[1] indicates the second element of the tuple, which is the information gain
    return best_attribute, best_gain

def build_decision_tree(mushrooms: list[Mushroom]):
    """
        Recursively Builds a decision tree using the provided mushroom dataset.

        Args:
            mushrooms (list[Mushroom]): List of Mushroom objects representing the dataset.

        Returns:
            Node: Root node of the decision tree.
        """
    if len(mushrooms) == 0:
        return None
    labels = [m.is_edible() for m in mushrooms]
    if labels.count(labels[0]) == len(labels): #checks if all the mushrooms are edible or not , .count() counts the number of edible or none edible mushrooms
        if labels[0]:
            return Node('Yes', is_leaf=True)
        else:
            return Node('No', is_leaf=True)
    best_attribute = choose_best_attribute(mushrooms)
    unique_values = []
    for m in mushrooms:
        if m.get_attribute(best_attribute[0]) not in unique_values:
           unique_values.append(m.get_attribute(best_attribute[0]))
    root = Node(best_attribute[0])

    for value in unique_values:
        sub_list = [m for m in mushrooms if m.get_attribute(best_attribute[0]) == value]
        child = build_decision_tree(sub_list) #creates a subtree
        root.add_edge(value, child)

    return root


def is_edible(root: Node, mushroom: Mushroom):
    """
        Determines if the provided mushroom is edible based on the attributes on the mushroom.

        Args:
            root (Node): Root node of the decision tree.
            mushroom (Mushroom): Mushroom object to classify.

        Returns:
            bool: True if the mushroom is classified as edible, False otherwise.

        """
    if root.is_leaf():
        return root.criterion_ != 'No'

    attribute_value = mushroom.get_attribute(root.criterion_)
    for child in root.edges_:
        if child.label_ == attribute_value:
            return is_edible(child.child_, mushroom)

    return False

def display(node, indent=''):
    """
         Recursively Displays the nodes off the decision tree starting from the specified node to in the end
         display the whole tree.

        Args:
            node (Node): Starting node of the decision tree.
            indent (str): Indentation string for visual representation.

        Returns:
            None
        """
    if node.is_leaf():
        print(indent + "Leaf: edible =", node.criterion_)
    else:
        print(indent + "Node: Criterion =", node.criterion_)

        for edge in node.edges_:
            print(indent + "|- " + edge.label_ + " -> ", end='')

            display(edge.child_, indent + "   ")


def decision_tree_to_boolean_expression(root: Node):
    """
       Converts the decision tree into a boolean expression.

       Args:
           root (Node): Root node of the decision tree.

       Returns:
           str: Boolean expression representing the decision tree.
       """
    conditions = []
    criteria = root.criterion_
    for edge in root.edges_:
        if edge.child_.is_leaf() and edge.child_.criterion_ == 'Yes':
            conditions.append(f"({criteria} = {edge.label_})")
        elif not edge.child_.is_leaf():
            conditions.append(f"({criteria}={edge.label_} AND ( \n  { decision_tree_to_boolean_expression(edge.child_)})")

    return " OR ".join(conditions)



def display_boolean_expression(expression):
    print("Boolean Expression:", expression)


def to_python(node, path):
    """
       Generates Python code to reproduce the decision tree logic and saves it to a file.

       Args:
           node (Node): Root node of the decision tree.
           path (str): File path to save the generated Python code.

       Returns:
           None
       """
    with open(path, 'w') as file:
        file.write("def predict(mushroom):\n")
        write_conditions(node, file, indent_level=1)

def write_conditions(node, file, indent_level):
    """
        Recursively writes conditional statements to a file based on the decision tree structure.

        Args:
            node (Node): Current node in the decision tree.
            file (file object): File object to write the conditional statements.
            indent_level (int): Current indentation level for proper formatting.

        Returns:
            None
        """
    if node.is_leaf():
        file.write(" " * (4 * indent_level) + f"return '{node.criterion_}'\n")
    else:
        file.write(" " * (4 * indent_level) + f"if mushroom['{node.criterion_}'] == '{node.edges_[0].label_}':\n")
        write_conditions(node.edges_[0].child_, file, indent_level + 1)
        for edge in node.edges_[1:]:
            file.write(" " * (4 * indent_level) + f"elif mushroom['{node.criterion_}'] == '{edge.label_}':\n")
            write_conditions(edge.child_, file, indent_level + 1)

def main():
    try:
        mushrooms = load_dataset('mushrooms.csv')
        tree_root = build_decision_tree(mushrooms)
        display(tree_root)
    except FileNotFoundError:
           print("File Error")

if __name__ == "__main__":
    main()


