mod controller;
mod coppeliasimulation;
mod collect_simulation_data;

use crate::coppeliasimulation::SimpleSceneWithEasyObstacles;

fn linear_continuos_to_discrete(y: f64, limit: f64, space: f64) -> usize {
    // y = x*a +b
    // x =0 => y = -5.8
    // x =14 => y = 5.8
    // 5.8 = a*14 -5.8
    //a = (5.8+5.8)/14
    // x = (y -b)/a
    // x = (pos -(-5.8))/(14/4.95)

    let a = limit * 2.0 / space;
    let x = (y + limit) / a;

    assert!(
        0.0 <= x && x <= space,
        "Erro on pos: {} get x: {}, space: {}",
        y,
        x,
        space
    );
    x.round() as usize
}

fn discrete_to_linear_continuos(x: usize, limit: f64, space: f64) -> f64 {
    let a = limit * 2.0 / space;

    let x = x as f64;

    let y = a * x - limit;

    y
}

#[test]
fn test_pos_continuos_to_discrete() {
    let space = 14.0;

    let a_pos = 5.8 * 2.0 / space;
    let b_pos = -5.8;

    let a_ori = 360f64.to_radians() / space;
    let b_ori = -180f64.to_radians();

    for x in 0..15 {
        let x = x as f64;
        let y_pos = a_pos * x + b_pos;
        let y_ori = a_ori * x + b_ori;
        let pred_pos = linear_continuos_to_discrete(y_pos, 5.8, space);
        let pred_ori = linear_continuos_to_discrete(y_ori, 180f64.to_radians(), space);
        assert_eq!(
            pred_pos, x as usize,
            "x:{} pred: {} y_pos: {}",
            x, pred_pos, y_pos
        );
        assert_eq!(pred_ori, x as usize, "x:{} pred: {}", x, pred_ori);
    }
}

fn main() {
   
    collect_simulation_data::collect_data();

}
