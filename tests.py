import unittest
from project import *

class TestMushroomDataLoading(unittest.TestCase):
    def setUp(self):
        self.mushrooms = load_dataset('mushrooms.csv')

    def test_load_dataset(self):
        m1 = self.mushrooms[0]
        self.assertFalse(m1.is_edible(), "Le premier champignon devrait être non comestible.")
        self.assertEqual(m1.get_attribute('cap-shape'), 'Convex')
        self.assertEqual(m1.get_attribute('odor'), 'Pungent')

        m2 = self.mushrooms[1]
        self.assertTrue(m2.is_edible(), "Le deuxième champignon devrait être comestible.")
        self.assertEqual(m2.get_attribute('cap-color'), 'Yellow')
        self.assertEqual(m2.get_attribute('odor'), 'Almond')

        m3 = self.mushrooms[2]
        self.assertTrue(m3.is_edible(), "Le troisième champignon devrait être comestible.")
        self.assertEqual(m3.get_attribute('cap-shape'), 'Bell')
        self.assertEqual(m3.get_attribute('odor'), 'Anise')

    def test_load_type(self):
        """
        This test verifies that the data has the correct types.

        """
        for mushroom in self.mushrooms:
            self.assertTrue(all(isinstance(value, str) for value in mushroom.attributes.values()))
            self.assertIsInstance(mushroom.edible, bool)

    def test_missing_value(self):
        """
          This test verifies that we do not have any missing data.

        """
        for mushroom in self.mushrooms:
            self.assertFalse(any(value is None for value in mushroom.attributes.values()))

    def test_node_creation(self):
        """
            test that verifies that if the class node is correct

        """

        parent_node = Node("odor")
        child_node = Node("Almond", is_leaf=True)


        parent_node.add_edge("Almond", child_node)
        self.assertEqual(parent_node.criterion_, "odor", "Le critère du nœud n'est pas correctement initialisé.")
        self.assertFalse(parent_node.is_leaf(), "Le nœud devrait être un nœud de division, pas une feuille.")
        self.assertEqual(len(parent_node.edges_), 1, "L'arête n'a pas été ajoutée avec succès à la liste des arêtes.")
        self.assertEqual(parent_node.edges_[0].label_, "Almond", "Le libellé de l'arête n'est pas correct.")
        self.assertEqual(parent_node.edges_[0].child_, child_node, "Le nœud enfant de l'arête n'est pas correct.")

    def test_calculate_information_gain(self):
        gain_1 = calculate_information_gain_for_attribute(self.mushrooms, 'odor')
        gain_2 = calculate_information_gain_for_attribute(self.mushrooms, 'spore-print-color')
        self.assertAlmostEqual(gain_1, 0.9060749773839998, places=5)
        self.assertAlmostEqual(gain_2, 0.4807049176849155, places=5)


def make_mushroom(attributes):
    ret = Mushroom(None)
    for k, v in attributes.items():
        ret.add_attribute(k, v)
    return ret

class TestBuildTree(unittest.TestCase):
    def setUp(self):
        self.test_tree_root = build_decision_tree(load_dataset('mushrooms.csv'))

    def test_tree_main_attribute(self):
        self.assertEqual(self.test_tree_root.criterion_, 'odor', "Le premier critère de division doit être 'odor'")
        nos = ['Pungent', 'Creosote', 'Foul', 'Fishy', 'Spicy', 'Musty']
        odors = {edge.label_: edge.child_ for edge in self.test_tree_root.edges_}
        for odor in nos:
            self.assertTrue(
                odors[odor].is_leaf() and odors[odor].criterion_ == 'No',
                f'Les champignons avec une odeur \'{odor}\' doivent être non-comestibles'
            )
    def test_tree_second_attribute(self):
        """
                Test verifying a second attribute for splitting the tree.

        """
        self.assertEqual(self.test_tree_root.criterion_, 'odor', "Le premier critère de division doit être 'odor'")
        nos = ['None']
        odors = {edge.label_: edge.child_ for edge in self.test_tree_root.edges_}
        for odor in nos:
            self.assertTrue(
                 odors[odor].criterion_ == 'spore-print-color',
                f'Les champignons avec une odeur \'{odor}\' doivent être non-comestibles'
            )
            pos = ['Brown','Black']
            spore_print_colors = {edge.label_: edge.child_ for edge in odors[odor].edges_}
            for spc in pos:
                self.assertTrue(
                    spore_print_colors[spc].criterion_ == 'Yes',
                    f'Les champignons avec une odeur \'{odor}\' et une couleur d\'impression des spores \'{spc}\' doivent être comestible'
                )
    def test_tree_prediction(self):
        root = self.test_tree_root
        self.assertTrue(is_edible(root, make_mushroom({'odor': 'Almond'})))
        self.assertFalse(is_edible(root, make_mushroom({'odor': 'None', 'spore-print-color': 'Green'})))
class TestDecisionTreeToBooleanExpression(unittest.TestCase):
    """

    Test for generating a simple tree with only one division criterion.

    """
    def test_simple_tree(self):

        root = Node("odor")
        edge1 = Node("Yes", is_leaf=True)
        edge2 = Node("No", is_leaf=True)
        root.add_edge("Almond", edge1)
        root.add_edge("Anise", edge2)


        expression = decision_tree_to_boolean_expression(root)


        self.assertEqual(expression, "(odor = Almond)")

    def test_complex_tree(self):
        """
            Test for generating a complex tree with multiple division criteria.

        """
        root = Node("odor")
        edge1 = Node("No", is_leaf=True)
        edge2 = Node("None")
        edge3 = Node("Yes", is_leaf=True)
        edge4 = Node("Yes", is_leaf=True)
        root.add_edge("Fishy", edge1)
        root.add_edge("None", edge2)
        edge2.add_edge("Brown", edge3)
        edge2.add_edge("Black", edge4)


        expression = decision_tree_to_boolean_expression(root)


        expected_expression = '(odor=None AND ( \n  (None = Brown) OR (None = Black))'
        self.assertEqual(expected_expression, expression)


if __name__ == '__main__':
    unittest.main()
