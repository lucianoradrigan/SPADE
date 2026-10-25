within ;
package SPADE "Modelica twins of SPADE's native plants (round PLECS/OM, route b). MIT licence."
  extends Modelica.Icons.Package;
  import SI = Modelica.Units.SI;

  package Components "Building blocks with no exact MSL equivalent"
    extends Modelica.Icons.Package;
    import SI = Modelica.Units.SI;

    model PolynomialStaticLoad
      "Load torque of GEM's PolynomialStaticLoad (src/driveflow/sim/loads.py:470-473, a/b/c and tau_decay from :430, :444-447)"
      extends Modelica.Mechanics.Rotational.Interfaces.PartialTorque;
      parameter SI.Torque a = 0.01 "Constant (Coulomb-like) term, datagen/runner.py:127";
      parameter Real b(unit="N.m.s/rad") = 0.05 "Linear (viscous) term, datagen/runner.py:127";
      parameter Real c(unit="N.m.s2/rad2") = 0.0 "Quadratic term, datagen/runner.py:127";
      parameter SI.Inertia jTotal = 0.0275 "J_rotor + J_load: the native linearises 'a' near zero speed with it";
      parameter SI.Time tauDecay = 1e-3 "loads.py:430";
      SI.AngularVelocity w "Shaft speed";
      SI.Torque tau "Load torque (opposes motion)";
    protected
      parameter SI.AngularVelocity wLim = a/jTotal*tauDecay "loads.py:447";
    equation
      w = der(phi);
      tau = c*sign(w)*w^2 + b*w + (if abs(w) > wLim then a*sign(w) else jTotal/tauDecay*w);
      flange.tau = tau "the load torque opposes motion: accelerating torque = -tau (MSL PartialTorque convention)";
    end PolynomialStaticLoad;
  end Components;

  model BuckAveraged "Synchronous buck, AVERAGED (sim/buck_system.py, buck_v1). Input duty, outputs v_out, i_L"
    extends Modelica.Icons.Example;
    parameter SI.Voltage Vin = 48.0 "sim/buck_system.py:41 (BuckParams.v_in_v)";
    parameter SI.Inductance L = 220e-6 "sim/buck_system.py:42 (l_h)";
    parameter SI.Capacitance C = 220e-6 "sim/buck_system.py:43 (c_f)";
    parameter SI.Resistance rL = 0.05 "sim/buck_system.py:44 (r_l_ohm)";
    parameter SI.Resistance R = 6.0 "sim/buck_system.py:45 (load_resistance_ohm)";
    Modelica.Blocks.Interfaces.RealInput duty "Duty cycle in [0, 1] (clipped, as BuckSystem.simulate)";
    Modelica.Blocks.Interfaces.RealOutput v_out(unit="V") "Output (capacitor) voltage";
    Modelica.Blocks.Interfaces.RealOutput i_L(unit="A") "Inductor current";
  protected
    Modelica.Blocks.Nonlinear.Limiter clip(uMax=1, uMin=0);
    Modelica.Electrical.Analog.Sources.SignalVoltage avgSwitchNode "d*Vin: the averaged switch node";
    Modelica.Electrical.Analog.Basic.Resistor resL(R=rL);
    Modelica.Electrical.Analog.Basic.Inductor ind(L=L, i(start=0, fixed=true));
    Modelica.Electrical.Analog.Basic.Capacitor cap(C=C, v(start=0, fixed=true));
    Modelica.Electrical.Analog.Basic.Resistor load(R=R);
    Modelica.Electrical.Analog.Basic.Ground gnd;
  equation
    connect(duty, clip.u);
    avgSwitchNode.v = clip.y*Vin;
    connect(avgSwitchNode.p, resL.p);
    connect(resL.n, ind.p);
    connect(ind.n, cap.p);
    connect(cap.p, load.p);
    connect(cap.n, gnd.p);
    connect(load.n, gnd.p);
    connect(avgSwitchNode.n, gnd.p);
    v_out = cap.v;
    i_L = ind.i;
  end BuckAveraged;

  model BuckSwitching
    "Synchronous buck with ideal switches (MSL PowerConverters half bridge) and an MSL PWM. One switching period = the native control step (20 us, 50 kHz)"
    extends Modelica.Icons.Example;
    parameter SI.Voltage Vin = 48.0 "sim/buck_system.py:41";
    parameter SI.Inductance L = 220e-6 "sim/buck_system.py:42";
    parameter SI.Capacitance C = 220e-6 "sim/buck_system.py:43";
    parameter SI.Resistance rL = 0.05 "sim/buck_system.py:44";
    parameter SI.Resistance R = 6.0 "sim/buck_system.py:45";
    parameter SI.Frequency fSwitch = 1/2e-5 "1/TAU, sim/buck_system.py:36";
    Modelica.Blocks.Interfaces.RealInput duty "Duty cycle in [0, 1]";
    Modelica.Blocks.Interfaces.RealOutput v_out(unit="V");
    Modelica.Blocks.Interfaces.RealOutput i_L(unit="A");
  protected
    Modelica.Blocks.Nonlinear.Limiter clip(uMax=1, uMin=0);
    Modelica.Electrical.PowerConverters.DCDC.Control.SignalPWM pwm(useConstantDutyCycle=false, f=fSwitch);
    Modelica.Electrical.PowerConverters.DCAC.SinglePhase2Level leg "Two ideal switches: a synchronous buck leg";
    Modelica.Electrical.Analog.Sources.ConstantVoltage src(V=Vin);
    Modelica.Electrical.Analog.Basic.Resistor resL(R=rL);
    Modelica.Electrical.Analog.Basic.Inductor ind(L=L, i(start=0, fixed=true));
    Modelica.Electrical.Analog.Basic.Capacitor cap(C=C, v(start=0, fixed=true));
    Modelica.Electrical.Analog.Basic.Resistor load(R=R);
    Modelica.Electrical.Analog.Basic.Ground gnd;
  equation
    connect(duty, clip.u);
    connect(clip.y, pwm.dutyCycle);
    connect(pwm.fire, leg.fire_p);
    connect(pwm.notFire, leg.fire_n);
    connect(src.p, leg.dc_p);
    connect(src.n, leg.dc_n);
    connect(src.n, gnd.p);
    connect(leg.ac, resL.p);
    connect(resL.n, ind.p);
    connect(ind.n, cap.p);
    connect(cap.p, load.p);
    connect(cap.n, gnd.p);
    connect(load.n, gnd.p);
    v_out = cap.v;
    i_L = ind.i;
  end BuckSwitching;

  partial model PartialDcMotor "dc_perm_ex_v1: GEM DcPermanentlyExcitedMotor + PolynomialStaticLoad (datagen/runner.py:119-144)"
    parameter SI.Resistance Ra = 16e-3 "sim/motors/dc_permanently_excited_motor.py:50 (r_a)";
    parameter SI.Inductance La = 19e-6 "dc_permanently_excited_motor.py:51 (l_a)";
    parameter SI.MagneticFlux psi_e = 0.165 "dc_permanently_excited_motor.py:52 (psi_e)";
    parameter SI.Inertia Jr = 0.025 "dc_permanently_excited_motor.py:53 (j_rotor)";
    parameter SI.Inertia Jload = 0.0025 "datagen/runner.py:127 (j_load)";
    parameter SI.AngularVelocity wNominal = 300 "dc_permanently_excited_motor.py:55";
    parameter SI.Current IaNominal = 97 "dc_permanently_excited_motor.py:55";
    parameter SI.Torque loadA = 0.01 "Load constant term a, datagen/runner.py:127";
    parameter Real loadB(unit="N.m.s/rad") = 0.05 "Load linear term b, datagen/runner.py:127";
    parameter Real loadC(unit="N.m.s2/rad2") = 0.0 "Load quadratic term c, datagen/runner.py:127";
    Modelica.Blocks.Interfaces.RealOutput omega(unit="rad/s");
    Modelica.Blocks.Interfaces.RealOutput i_a(unit="A");
    Modelica.Blocks.Interfaces.RealOutput torque(unit="N.m") "Electromagnetic torque psi_e*i_a (native torque_nm)";
  protected
    Modelica.Electrical.Machines.BasicMachines.DCMachines.DC_PermanentMagnet dcpm(
      VaNominal=psi_e*wNominal + Ra*IaNominal, IaNominal=IaNominal, wNominal=wNominal, Ra=Ra, La=La, Jr=Jr, Js=Jr,
      TaNominal=293.15, TaRef=293.15, TaOperational=293.15, alpha20a=0, useSupport=false, useThermalPort=false,
      ia(fixed=true, start=0), wMechanical(fixed=true, start=0), phiMechanical(fixed=true, start=0))
      "VaNominal chosen so that the machine's flux constant equals psi_e exactly (no brush, core, friction or stray losses)";
    Modelica.Mechanics.Rotational.Components.Inertia loadInertia(J=Jload);
    Components.PolynomialStaticLoad loadTorque(a=loadA, b=loadB, c=loadC, jTotal=Jr + Jload);
    Modelica.Electrical.Analog.Basic.Ground gnd;
  equation
    connect(dcpm.flange, loadInertia.flange_a);
    connect(loadInertia.flange_b, loadTorque.flange);
    omega = dcpm.wMechanical;
    i_a = dcpm.ia;
    torque = dcpm.airGapDC.tauElectrical;
  end PartialDcMotor;

  model DcMotorAveraged "PM DC motor fed by a controlled voltage source. Input v_a. Native: converter u = duty*65 V (datagen/runner.py:141)"
    extends Modelica.Icons.Example;
    extends PartialDcMotor;
    Modelica.Blocks.Interfaces.RealInput v_a(unit="V") "Armature voltage";
  protected
    Modelica.Electrical.Analog.Sources.SignalVoltage supply;
  equation
    supply.v = v_a;
    connect(supply.p, dcpm.pin_ap);
    connect(supply.n, gnd.p);
    connect(dcpm.pin_an, gnd.p);
  end DcMotorAveraged;

  model DcMotorSwitching
    "PM DC motor fed by a one-quadrant step-down chopper (ideal switch + freewheeling diode, MSL PowerConverters) from 65 V; one PWM period = the native control step (100 us)"
    extends Modelica.Icons.Example;
    extends PartialDcMotor;
    parameter SI.Voltage Vsupply = 65.0 "datagen/runner.py:141 (IdealVoltageSupply u_nominal)";
    parameter SI.Frequency fSwitch = 1/1e-4 "1/TAU, datagen/runner.py:71";
    Modelica.Blocks.Interfaces.RealInput duty "Duty cycle in [0, 1]";
  protected
    Modelica.Blocks.Nonlinear.Limiter clip(uMax=1, uMin=0);
    Modelica.Electrical.PowerConverters.DCDC.Control.SignalPWM pwm(useConstantDutyCycle=false, f=fSwitch);
    Modelica.Electrical.PowerConverters.DCDC.ChopperStepDown chopper;
    Modelica.Electrical.Analog.Sources.ConstantVoltage src(V=Vsupply);
  equation
    connect(duty, clip.u);
    connect(clip.y, pwm.dutyCycle);
    connect(pwm.fire, chopper.fire_p);
    connect(src.p, chopper.dc_p1);
    connect(src.n, chopper.dc_n1);
    connect(src.n, gnd.p);
    connect(chopper.dc_p2, dcpm.pin_ap);
    connect(chopper.dc_n2, dcpm.pin_an) "the armature returns through the chopper only: grounding pin_an too over-determines dc_n1/dc_n2";
  end DcMotorSwitching;

  annotation (uses(Modelica(version="4.1.0")));
end SPADE;
