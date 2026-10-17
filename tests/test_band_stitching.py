"""Band stitching: overlapping band transcripts are joined at the shared words, with no repeated lines."""

from src.pipeline.band_tiler import stitch_band_transcripts


def test_partial_first_line_of_lower_band_is_not_repeated():
    # the lower band's top edge cuts "as magical by speaking of", so its first line is partial
    upper = ("A book is a friend.\nIt describes the book\nas magical by [struck: narating] speaking of\n"
             "the things a book can offer to its\nlovers.")
    lower = "as [truncated]\nthe things a book can offer to its\nlovers.\nIt is a good poem."
    out = stitch_band_transcripts([upper, lower])
    assert out.count("the things a book can offer") == 1
    assert "[truncated]" not in out
    assert out.endswith("lovers.\nIt is a good poem.")
    assert "as magical by [struck: narating] speaking of" in out


def test_cut_edges_on_both_sides_are_dropped():
    upper = ("g) Ans: Portia's father had a plan\nto choose her has husband was that\n"
             "all the suiter will have to choose\na casket from the three thuse[truncated]")
    lower = ("to choose her [truncated]\nall the suiter will have to choose\n"
             "a casket from the [struck: thare] three\ncaskets and the one who can rightly")
    out = stitch_band_transcripts([upper, lower])
    assert out == ("g) Ans: Portia's father had a plan\nto choose her has husband was that\n"
                   "all the suiter will have to choose\na casket from the [struck: thare] three\n"
                   "caskets and the one who can rightly")


def test_no_shared_text_keeps_both_bands_whole():
    upper = "Once there lived a king.\nHe was kind.\n[truncated]"
    lower = "The island was green.\nBirds sang every morning."
    out = stitch_band_transcripts([upper, lower])
    assert out == "Once there lived a king.\nHe was kind.\nThe island was green.\nBirds sang every morning."


def test_single_word_match_is_not_taken_for_the_overlap():
    # a lone "Ans:" in both bands is not evidence of overlap; nothing may be dropped
    upper = "c) Who was the king?\nAns:\nThe king was Lear."
    lower = "d) Why did he cry?\nAns:\nHe lost his daughter."
    out = stitch_band_transcripts([upper, lower])
    assert out == upper + "\n" + lower


def test_repeated_line_seams_at_the_overlap_not_the_earlier_copy():
    upper = ("the king was kind and wise\nevery one loved him\n"
             "the king was kind and wise\nhe ruled for many [truncated]")
    lower = "the king was kind and wise\nhe ruled for many years\nand then he died"
    out = stitch_band_transcripts([upper, lower])
    assert out == ("the king was kind and wise\nevery one loved him\n"
                   "the king was kind and wise\nhe ruled for many years\nand then he died")


def test_three_bands_chain():
    b1 = "line one is here\nline two is here\nline three is"
    b2 = "line two is here\nline three is here\nline four is here\nline five is"
    b3 = "line four is here\nline five is here\nline six is here"
    out = stitch_band_transcripts([b1, b2, b3])
    assert out.split("\n") == ["line one is here", "line two is here", "line three is here",
                               "line four is here", "line five is here", "line six is here"]


def test_single_and_empty_inputs():
    assert stitch_band_transcripts([]) == ""
    assert stitch_band_transcripts(["  only band  "]) == "only band"
    assert stitch_band_transcripts(["first band", ""]) == "first band"


def test_overlap_read_differently_by_the_two_bands_is_not_repeated():
    # real case (dev page): each band misread the overlap in its own way, so no complete line is shared
    # word for word; the near-identical lines are matched instead of being kept twice
    upper = ("himself visited the tree he says \"Why\nyou are refusing to cut the trees?\"\n"
             "The old tree says, \" The trees are not\nnormal troees they are special trees\n"
             "from the paradise. If you cut them")
    lower = ("norormal troees they are special troes\nfroom the parodise. If you cut them\n"
             "you'll get huge disfame you'll lose\neverything you have this is the last")
    out = stitch_band_transcripts([upper, lower])
    assert out == upper + "\nyou'll get huge disfame you'll lose\neverything you have this is the last"
