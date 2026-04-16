import astnc as at


def test_workpoints_are_registered():
    names = at.available_workpoints()
    assert names == ("l1", "l2", "l3")
    assert at.get_workpoint("l1").method_options["leaf_tol"] < at.get_workpoint("l3").method_options["leaf_tol"]

