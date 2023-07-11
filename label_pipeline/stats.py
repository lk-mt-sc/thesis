from statistics import mean


class Statistics():
    def __init__(self, gui_statistics):
        self.statistics = gui_statistics.create_statistics(n=20)
        self.reset_statistics()

    def reset_statistics(self):
        self.statistics[0][0].set('Data Points (D.P.) Labeled')
        self.statistics[1][0].set('D.P. Not Hidden and Correct')
        self.statistics[2][0].set('D.P. Hidden and Presumably Correct')
        self.statistics[3][0].set('D.P. Not Hidden and Incorrect')
        self.statistics[4][0].set('D.P. Hidden and Incorrect')
        self.statistics[5][0].set('D.P. Hidden and Presumably Incorrect')
        self.statistics[6][0].set('Avg. Dist. of Switching D.P. to Neighbors')
        self.statistics[7][0].set('Avg. Dist. of Not Switching D.P. to Neighbors')
        self.statistics[8][0].set('Avg. Conf. of Not Hidden Correct D.P.')
        self.statistics[9][0].set('Avg. Conf. of Hidden Presumably Correct D.P.')
        self.statistics[10][0].set('Avg. Conf. of Not Hidden Incorrect D.P.')
        self.statistics[11][0].set('Avg. Conf. of Hidden Incorrect D.P.')
        self.statistics[12][0].set('Avg. Conf. of Hidden Presumably Incorrect D.P.')
        self.statistics[13][0].set('Not Hidden Correct D.P. with a Confidence < 0.75')
        self.statistics[14][0].set('Hidden Pres. Correct D.P. with a Confidence < 0.75')
        self.statistics[15][0].set('Not Hidden Incorrect D.P. with a Confidence > 0.75')
        self.statistics[16][0].set('Hidden Incorrect D.P. with a Confidence > 0.75')
        self.statistics[17][0].set('Hidden Pres. Incorrect D.P. with a Confidence > 0.75')
        self.statistics[18][0].set('Bounding Boxes Cutting the Climber Off')
        self.statistics[19][0].set('Side Swaps')

        for i in range(len(self.statistics)):
            self.statistics[i][1].set('-')
            self.statistics[i][2].set('-')

    def as_percentage(self, x1, x2=None, decimals=2):
        if x2 is None:
            x2 = 1

        return f'{round(x1/x2 * 100, decimals)}%'

    def as_numbers(self, x1, x2):
        return f'{x1}/{x2}'

    # dp = data points, l = label, lbd = labeled data, cur = current
    def calculate_statistics(self, all_lbd, cur_lbd):
        self.reset_statistics()

        n_dp_all = sum(lbd.n_labels for lbd in all_lbd)
        n_dp_cur = cur_lbd.n_labels

        n_labeled_dp_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[0] != -1)
        n_labeled_dp_cur = sum(1 for l in cur_lbd.labels if l[0] != -1)
        self.statistics[0][1].set(self.as_percentage(n_labeled_dp_all, n_dp_all))
        self.statistics[0][2].set(self.as_numbers(n_labeled_dp_cur, n_dp_cur))

        n_dp_not_hidden_correct_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[0] == 0)
        n_dp_not_hidden_correct_cur = sum(1 for l in cur_lbd.labels if l[0] == 0)
        self.statistics[1][1].set(self.as_percentage(n_dp_not_hidden_correct_all, n_dp_all))
        self.statistics[1][2].set(self.as_numbers(n_dp_not_hidden_correct_cur, n_dp_cur))

        n_dp_hidden_presumably_correct_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[0] == 1)
        n_dp_hidden_presumably_correct_cur = sum(1 for l in cur_lbd.labels if l[0] == 1)
        self.statistics[2][1].set(self.as_percentage(n_dp_hidden_presumably_correct_all, n_dp_all))
        self.statistics[2][2].set(self.as_numbers(n_dp_hidden_presumably_correct_cur, n_dp_cur))

        n_dp_not_hidden_incorrect_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[0] == 2)
        n_dp_not_hidden_incorrect_cur = sum(1 for l in cur_lbd.labels if l[0] == 2)
        self.statistics[3][1].set(self.as_percentage(n_dp_not_hidden_incorrect_all, n_dp_all))
        self.statistics[3][2].set(self.as_numbers(n_dp_not_hidden_incorrect_cur, n_dp_cur))

        n_dp_hidden_incorrect_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[0] == 3)
        n_dp_hidden_incorrect_cur = sum(1 for l in cur_lbd.labels if l[0] == 3)
        self.statistics[4][1].set(self.as_percentage(n_dp_hidden_incorrect_all, n_dp_all))
        self.statistics[4][2].set(self.as_numbers(n_dp_hidden_incorrect_cur, n_dp_cur))

        n_dp_hidden_presumably_incorrect_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[0] == 4)
        n_dp_hidden_presumably_incorrect_cur = sum(1 for l in cur_lbd.labels if l[0] == 4)
        self.statistics[5][1].set(self.as_percentage(n_dp_hidden_presumably_incorrect_all, n_dp_all))
        self.statistics[5][2].set(self.as_numbers(n_dp_hidden_presumably_incorrect_cur, n_dp_cur))

        confidence_not_hidden_correct_dp_all = []
        confidence_hidden_presumably_correct_dp_all = []
        confidence_not_hidden_incorrect_dp_all = []
        confidence_hidden_incorrect_dp_all = []
        confidence_hidden_presumably_incorrect_dp_all = []

        for lbd in all_lbd:
            for i, l in enumerate(lbd.labels):
                if l[0] == 0:
                    confidence_not_hidden_correct_dp_all.append(lbd.scores[i])
                if l[0] == 1:
                    confidence_hidden_presumably_correct_dp_all.append(lbd.scores[i])
                if l[0] == 2:
                    confidence_not_hidden_incorrect_dp_all.append(lbd.scores[i])
                if l[0] == 3:
                    confidence_hidden_incorrect_dp_all.append(lbd.scores[i])
                if l[0] == 4:
                    confidence_hidden_presumably_incorrect_dp_all.append(lbd.scores[i])

        if confidence_not_hidden_correct_dp_all:
            avg_confidence_not_hidden_correct_dp_all = mean(confidence_not_hidden_correct_dp_all)
            self.statistics[8][1].set(self.as_percentage(avg_confidence_not_hidden_correct_dp_all))
        if confidence_hidden_presumably_correct_dp_all:
            avg_confidence_hidden_presumably_correct_dp_all = mean(confidence_hidden_presumably_correct_dp_all)
            self.statistics[9][1].set(self.as_percentage(avg_confidence_hidden_presumably_correct_dp_all))
        if confidence_not_hidden_incorrect_dp_all:
            avg_confidence_not_hidden_incorrect_dp_all = mean(confidence_not_hidden_incorrect_dp_all)
            self.statistics[10][1].set(self.as_percentage(avg_confidence_not_hidden_incorrect_dp_all))
        if confidence_hidden_incorrect_dp_all:
            avg_confidence_hidden_incorrect_dp_all = mean(confidence_hidden_incorrect_dp_all)
            self.statistics[11][1].set(self.as_percentage(avg_confidence_hidden_incorrect_dp_all))
        if confidence_hidden_presumably_incorrect_dp_all:
            avg_confidence_hidden_presumably_incorrect_dp_all = mean(confidence_hidden_presumably_incorrect_dp_all)
            self.statistics[12][1].set(self.as_percentage(avg_confidence_hidden_presumably_incorrect_dp_all))

        confidence_not_hidden_correct_dp_cur = []
        confidence_hidden_presumably_correct_dp_cur = []
        confidence_not_hidden_incorrect_dp_cur = []
        confidence_hidden_incorrect_dp_cur = []
        confidence_hidden_presumably_incorrect_dp_cur = []

        for i, l in enumerate(cur_lbd.labels):
            if l[0] == 0:
                confidence_not_hidden_correct_dp_cur.append(cur_lbd.scores[i])
            if l[0] == 1:
                confidence_hidden_presumably_correct_dp_cur.append(cur_lbd.scores[i])
            if l[0] == 2:
                confidence_not_hidden_incorrect_dp_cur.append(cur_lbd.scores[i])
            if l[0] == 3:
                confidence_hidden_incorrect_dp_cur.append(cur_lbd.scores[i])
            if l[0] == 4:
                confidence_hidden_presumably_incorrect_dp_cur.append(cur_lbd.scores[i])

        if confidence_not_hidden_correct_dp_cur:
            avg_confidence_not_hidden_correct_dp_cur = mean(confidence_not_hidden_correct_dp_cur)
            self.statistics[8][2].set(self.as_percentage(avg_confidence_not_hidden_correct_dp_cur))
        if confidence_hidden_presumably_correct_dp_cur:
            avg_confidence_hidden_presumably_correct_dp_cur = mean(confidence_hidden_presumably_correct_dp_cur)
            self.statistics[9][2].set(self.as_percentage(avg_confidence_hidden_presumably_correct_dp_cur))
        if confidence_not_hidden_incorrect_dp_cur:
            avg_confidence_not_hidden_incorrect_dp_cur = mean(confidence_not_hidden_incorrect_dp_cur)
            self.statistics[10][2].set(self.as_percentage(avg_confidence_not_hidden_incorrect_dp_cur))
        if confidence_hidden_incorrect_dp_cur:
            avg_confidence_hidden_incorrect_dp_cur = mean(confidence_hidden_incorrect_dp_cur)
            self.statistics[11][2].set(self.as_percentage(avg_confidence_hidden_incorrect_dp_cur))
        if confidence_hidden_presumably_incorrect_dp_cur:
            avg_confidence_hidden_presumably_incorrect_dp_cur = mean(confidence_hidden_presumably_incorrect_dp_cur)
            self.statistics[12][2].set(self.as_percentage(avg_confidence_hidden_presumably_incorrect_dp_cur))

        x = 0.8
        n_dp_not_hidden_correct_less_x_confidence_all = sum(1 for c in confidence_not_hidden_correct_dp_all if c < x)
        n_dp_not_hidden_correct_less_x_confidence_cur = sum(1 for c in confidence_not_hidden_correct_dp_cur if c < x)
        self.statistics[13][1].set(self.as_percentage(n_dp_not_hidden_correct_less_x_confidence_all, n_dp_all))
        self.statistics[13][2].set(self.as_numbers(n_dp_not_hidden_correct_less_x_confidence_cur, n_dp_cur))

        n_dp_hidden_presumably_correct_less_x_confidence_all = sum(
            1 for c in confidence_hidden_presumably_correct_dp_all if c < x)
        n_dp_hidden_presumably_correct_less_x_confidence_cur = sum(
            1 for c in confidence_hidden_presumably_correct_dp_cur if c < x)
        self.statistics[14][1].set(self.as_percentage(n_dp_hidden_presumably_correct_less_x_confidence_all, n_dp_all))
        self.statistics[14][2].set(self.as_numbers(n_dp_hidden_presumably_correct_less_x_confidence_cur, n_dp_cur))

        n_dp_not_hidden_incorrect_greater_x_confidence_all = sum(
            1 for c in confidence_not_hidden_incorrect_dp_all if c > x)
        n_dp_not_hidden_incorrect_greater_x_confidence_cur = sum(
            1 for c in confidence_not_hidden_incorrect_dp_cur if c > x)
        self.statistics[15][1].set(self.as_percentage(n_dp_not_hidden_incorrect_greater_x_confidence_all, n_dp_all))
        self.statistics[15][2].set(self.as_numbers(n_dp_not_hidden_incorrect_greater_x_confidence_cur, n_dp_cur))

        n_dp_hidden_incorrect_greater_x_confidence_all = sum(1 for c in confidence_hidden_incorrect_dp_all if c > x)
        n_dp_hidden_incorrect_greater_x_confidence_cur = sum(1 for c in confidence_hidden_incorrect_dp_cur if c > x)
        self.statistics[16][1].set(self.as_percentage(n_dp_hidden_incorrect_greater_x_confidence_all, n_dp_all))
        self.statistics[16][2].set(self.as_numbers(n_dp_hidden_incorrect_greater_x_confidence_cur, n_dp_cur))

        n_dp_hidden_presumably_incorrect_greater_x_confidence_all = sum(
            1 for c in confidence_hidden_presumably_incorrect_dp_all if c > x)
        n_dp_hidden_presumably_incorrect_greater_x_confidence_cur = sum(
            1 for c in confidence_hidden_presumably_incorrect_dp_cur if c > x)
        self.statistics[17][1].set(self.as_percentage(
            n_dp_hidden_presumably_incorrect_greater_x_confidence_all, n_dp_all))
        self.statistics[17][2].set(self.as_numbers(
            n_dp_hidden_presumably_incorrect_greater_x_confidence_cur, n_dp_cur))

        n_side_swaps_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[2])
        n_side_swaps_cur = sum(1 for l in cur_lbd.labels if l[2])
        self.statistics[18][1].set(self.as_numbers(n_side_swaps_all, n_dp_all))
        self.statistics[18][2].set(self.as_numbers(n_side_swaps_cur, n_dp_cur))

        n_bb_cuts_climber_all = sum(1 for lbd in all_lbd for l in lbd.labels if l[3])
        n_bb_cuts_climber_cur = sum(1 for l in cur_lbd.labels if l[3])
        self.statistics[19][1].set(self.as_numbers(n_bb_cuts_climber_all, n_dp_all))
        self.statistics[19][2].set(self.as_numbers(n_bb_cuts_climber_cur, n_dp_cur))

    def calculate_high_pass_properties(self, labeled_data):
        # WIP in external script
        pass
