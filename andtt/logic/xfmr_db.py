from andtt.logic import DatabaseFast

class XMFRDatabaseFast(DatabaseFast):

    def __init__(self, datalog):
        super(XMFRDatabaseFast, self).__init__(datalog)

    def make_transform_idx_and_dim(self, fact, transform_idx_and_dim, datalog):
        """
        each fact has form : term, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        temp = [
            fact[-2], fact[-1] # betaparam and mparam
        ]
        # get dim and zero
        in_dim, in_zero = self.aug_dim( fact[1:-3], datalog )
        out_dim, out_zero = self.aug_dim( fact[:1], datalog )
        head_functor = datalog.get_functor(fact[0])
        # store all info
        temp += [
            in_dim, in_zero,
            out_dim, out_zero,
            head_functor,
            # datalog.get_functor(fact[0]) in datalog.auxilary['event_functors']
            head_functor in datalog.auxilary['event_functors']
            # event or not (i.e. intensity exist)
        ]
        transform_idx_and_dim.add(tuple(temp))

    def make_see_idx_and_dim(self, fact, see_idx_and_dim, datalog):
        """
        each fact has form : term, event, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        temp = [
            fact[-2], fact[-1] # betaparam and mparam
        ]
        # get dim and zero
        in_dim, in_zero = self.aug_dim( fact[1:-3], datalog )
        out_dim, out_zero = self.aug_dim( fact[:1], datalog )
        event_dim, _ = self.aug_dim([fact[1], ], datalog)
        head_functor = datalog.get_functor(fact[0])
        # store all info
        temp += [
            in_dim, in_zero,
            out_dim, out_zero,
            # add extra info for event_dim -- we need to use that when defining neural network
            event_dim,
            head_functor,
            head_functor in datalog.auxilary['event_functors']
            # event or not (i.e. intensity exist)
        ]
        see_idx_and_dim.add(tuple(temp))
