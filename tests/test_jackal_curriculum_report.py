from navigation.jackal_curriculum_report import summarize, wilson


def test_wilson_exposes_small_sample_uncertainty():
    assert wilson(0,0) is None
    low,high=wilson(0,4)
    assert low==0 and .48<high<.50


def test_partial_episodes_do_not_count_as_outcomes():
    result=summarize([dict(status='partial',success=True),dict(status='completed',success=False,
        collision=False,timeout=True,episode_return=2.,length=300)])
    assert result['successes']==0 and result['completed']==1
    assert result['partial_or_failed']==1 and result['mean_length']==300
