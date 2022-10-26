use std::io::stdin;
use std::io::stdout;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

pub struct ManualController;

impl ManualController {
    pub fn get_velocity() -> Option<Vec<f64>> {
        let std_in = stdin();
        let out = stdout().into_raw_mode().unwrap();

        let key = std_in.keys().next().unwrap().unwrap();

        out.suspend_raw_mode().unwrap();

        match key {
            Key::Left => Some(vec![2.0, 0.0]),
            Key::Right => Some(vec![0.0, 2.0]),
            Key::Up => Some(vec![2.0, 2.0]),
            Key::Down => Some(vec![-2.0, -2.0]),

            Key::Ctrl('c') => None,

            _ => None,
        }
    }
}
