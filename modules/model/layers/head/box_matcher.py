import torch
from torch import Tensor

class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the BxMxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size BxN containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): a BxMxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): a BxN tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            raise ValueError("No match quality matrix provided")

        B, M, N = match_quality_matrix.shape
        matches = torch.full((B, N), self.BELOW_LOW_THRESHOLD, dtype=torch.int64, device=match_quality_matrix.device)

        for b in range(B):
            matched_vals, match_indices = match_quality_matrix[b].max(dim=0)
            if self.allow_low_quality_matches:
                all_matches = match_indices.clone()
            else:
                all_matches = None  # type: ignore[assignment]

            below_low_threshold = matched_vals < self.low_threshold
            between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
            match_indices[below_low_threshold] = self.BELOW_LOW_THRESHOLD
            match_indices[between_thresholds] = self.BETWEEN_THRESHOLDS
            matches[b] = match_indices

            if self.allow_low_quality_matches and all_matches is not None:
                self.set_low_quality_matches_(matches[b], all_matches, match_quality_matrix[b])

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class SSDMatcher(Matcher):
    def __init__(self, threshold: float) -> None:
        super().__init__(threshold, threshold, allow_low_quality_matches=False)

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        matches = super().__call__(match_quality_matrix)

        B, M, N = match_quality_matrix.shape
        for b in range(B):
            max_quality_matrix, highest_quality_pred_foreach_gt = match_quality_matrix[b].max(dim=1)
            highest_quality_pred_foreach_gt = highest_quality_pred_foreach_gt[max_quality_matrix > 0]
            matches[b, highest_quality_pred_foreach_gt] = torch.arange(
                highest_quality_pred_foreach_gt.size(0), dtype=torch.int64, device=highest_quality_pred_foreach_gt.device
            )

        return matches
